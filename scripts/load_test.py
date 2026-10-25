"""Load the model through ai-toolkit's engine and render one image (no HTTP).

    python scripts/load_test.py --size 512 --steps 8

Useful to validate a fresh checkout or a new quantisation/offload setting.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_server.config import load_config  # noqa: E402
from moss_server.engine import GenerationRequest, MossEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MOSS_MODEL_DIR", ""), help="MOSS-Image diffusers bundle")
    parser.add_argument("--aitk-root", default=os.environ.get("AI_TOOLKIT_DIR", ""), help="ai-toolkit checkout")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--prompt", default="a photo of a red apple on a wooden table")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="append", default=None)
    parser.add_argument("--lora-scale", type=float, default=0.7)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--out", default="outputs/load_test.png")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    overrides = {
        "model_dir": args.model_dir,
        "aitk_root": args.aitk_root,
        "warmup": False,
        "lora_scale": args.lora_scale,
    }
    if args.lora:
        overrides["loras"] = args.lora
    if args.no_quantize:
        overrides["quantize"] = False
        overrides["quantize_te"] = False
    config = load_config(None, overrides).resolve(Path.cwd()).validate()

    engine = MossEngine(config)
    started = time.perf_counter()
    engine.ensure_loaded()
    print(f"\nload: {time.perf_counter() - started:.1f}s")
    for key, value in engine.describe().items():
        if key in ("model_meta", "stats", "loras", "active"):
            continue
        print(f"  {key}: {value}")

    started = time.perf_counter()
    images = engine.generate(
        GenerationRequest(
            prompt=args.prompt,
            height=args.size,
            width=args.size,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
    )
    elapsed = time.perf_counter() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].image.save(out)
    print(f"\nGenerated {images[0].image.size} in {elapsed:.1f}s -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
