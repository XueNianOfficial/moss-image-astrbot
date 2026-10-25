"""End-to-end smoke test against a running server.

    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --size 320 --steps 2

Exercises the lazy lifecycle (idle -> load on demand -> unload), the OpenAI and
native endpoints, SSE streaming and (optionally) LoRA load/scale/unload.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def call(method: str, url: str, body: dict | None = None, api_key: str = "", stream: bool = False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        response = urllib.request.urlopen(request, timeout=1800)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {url} -> {error.code}: {error.read().decode('utf-8', 'replace')[:500]}")
    if stream:
        return response
    return json.loads(response.read().decode("utf-8"))


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name} {detail}")
    if not condition:
        raise SystemExit(f"smoke test failed at: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lora", default=None, help="optional LoRA path to load/unload")
    parser.add_argument("--skip-admin", action="store_true", help="skip the unload/load round-trip")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("health")
    deadline = time.time() + 300
    while True:
        health = call("GET", f"{base}/health")
        if health.get("status") in ("ready", "idle"):
            break
        if time.time() > deadline:
            raise SystemExit(f"server not usable: {health}")
        time.sleep(2)
    check("healthy", health["status"] in ("ready", "idle"), f"state={health['status']}")

    print("status")
    status = call("GET", f"{base}/v1/status", api_key=args.api_key)
    check("engine reported", status.get("engine") == "ai-toolkit", str(status.get("engine")))
    check("quantisation reported", status.get("quantize") is not None, str(status.get("qtype")))
    check("lazy_load flag", "lazy_load" in status, str(status.get("lazy_load")))
    check("idle policy", "idle_unload_seconds" in status, str(status.get("idle_unload_seconds")))

    print("openai images endpoint (loads the model if needed)")
    started = time.time()
    result = call(
        "POST",
        f"{base}/v1/images/generations",
        {"prompt": "a photo of a red apple", "size": f"{args.size}x{args.size}",
         "n": 1, "steps": args.steps, "guidance_scale": 3.0, "seed": 1234,
         "response_format": "b64_json"},
        api_key=args.api_key,
    )
    check("one image", len(result["data"]) == 1)
    raw = base64.b64decode(result["data"][0]["b64_json"])
    check("valid png", raw.startswith(PNG_MAGIC), f"{len(raw)} bytes in {time.time() - started:.1f}s")
    check("seed echoed", result["data"][0]["seed"] == 1234)

    print("native endpoint (url output)")
    result = call(
        "POST",
        f"{base}/v1/moss/generate",
        {"prompt": "a cat", "width": args.size, "height": args.size, "steps": args.steps,
         "guidance_scale": 1.0, "response_format": "url"},
        api_key=args.api_key,
    )
    url = result["data"][0]["url"]
    check("image url", url.startswith("http"), url)
    fetched = urllib.request.urlopen(url, timeout=60).read()
    check("url serves png", fetched.startswith(PNG_MAGIC), f"{len(fetched)} bytes")

    print("sse streaming")
    events = []
    with call(
        "POST",
        f"{base}/v1/moss/generate?stream=true",
        {"prompt": "a lighthouse", "width": args.size, "height": args.size, "steps": max(2, args.steps),
         "guidance_scale": 1.0, "response_format": "b64_json"},
        api_key=args.api_key,
        stream=True,
    ) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if line.startswith("data:") and events and events[-1] == "done":
                break
    check("status event", "status" in events, ",".join(events[:4]))
    check("progress events", "progress" in events, f"{len(events)} events")
    check("done event", events[-1] == "done")

    if args.lora:
        print("lora management")
        loaded = call("POST", f"{base}/v1/loras/load",
                      {"path": args.lora, "name": "smoke", "scale": 0.7}, api_key=args.api_key)
        check("lora loaded", loaded["layers"] > 0, f"{loaded['layers']} layers, rank {loaded['rank']}")
        listing = call("GET", f"{base}/v1/loras", api_key=args.api_key)
        check("lora listed", any(item["name"] == "smoke" for item in listing["loaded"]))
        call("POST", f"{base}/v1/loras/smoke/scale", {"scale": 0.5}, api_key=args.api_key)
        result = call("POST", f"{base}/v1/moss/generate",
                      {"prompt": "a fox", "size": f"{args.size}x{args.size}", "steps": args.steps,
                       "guidance_scale": 1.0, "loras": {"smoke": 0.5}, "response_format": "url"},
                      api_key=args.api_key)
        check("generation with lora", len(result["data"]) == 1)
        call("DELETE", f"{base}/v1/loras/smoke", api_key=args.api_key)
        listing = call("GET", f"{base}/v1/loras", api_key=args.api_key)
        check("lora unloaded", not any(item["name"] == "smoke" for item in listing["loaded"]))

    if not args.skip_admin:
        print("admin unload / load")
        unloaded = call("POST", f"{base}/v1/admin/unload", api_key=args.api_key)
        check("unloaded", unloaded.get("state") == "idle", str(unloaded))
        after = call("GET", f"{base}/v1/status")
        check("vram released", after.get("vram_used_gib") is None, str(after.get("state")))
        reloaded = call("POST", f"{base}/v1/admin/load", api_key=args.api_key)
        check("reloaded", reloaded.get("state") == "ready", f"{reloaded.get('seconds')}s")

        # Regression: a reload must rebuild every component the decode step needs.
        print("generation after reload")
        result = call(
            "POST",
            f"{base}/v1/images/generations",
            {"prompt": "a green field", "size": f"{args.size}x{args.size}", "steps": args.steps,
             "guidance_scale": 1.0, "seed": 5, "response_format": "b64_json"},
            api_key=args.api_key,
        )
        raw = base64.b64decode(result["data"][0]["b64_json"])
        check("png after reload", raw.startswith(PNG_MAGIC), f"{len(raw)} bytes")

    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
