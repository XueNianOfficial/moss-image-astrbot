"""Integration test for the AstrBot plugin against the running server.

Uses the real plugin class, the real HTTP client and the real AstrBot message
components, but a fake event and a fake LLM provider, so the running bot is not
touched.  Skips when AstrBot is not installed on this machine.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path

APP = Path(
    os.environ.get("ASTRBOT_APP_DIR")
    or (Path(os.environ.get("LOCALAPPDATA", "")) / "AstrBot" / "backend" / "app")
)
PLUGINS = Path(
    os.environ.get("ASTRBOT_PLUGINS_DIR")
    or (Path(os.environ.get("USERPROFILE", "")) / ".astrbot" / "data" / "plugins")
)
SERVER = os.environ.get("MOSS_SERVER_URL", "http://127.0.0.1:8000")

if not APP.is_dir() or not PLUGINS.is_dir():
    print("skip: AstrBot is not installed here (set ASTRBOT_APP_DIR / ASTRBOT_PLUGINS_DIR)")
    raise SystemExit(0)

# Importing astrbot initialises its data directory relative to the cwd, so run
# from a scratch directory instead of polluting the project.
os.chdir(tempfile.mkdtemp(prefix="astrbot-test-"))

for entry in (str(APP), str(PLUGINS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from astrbot.api.message_components import Plain  # noqa: E402

from astrbot_plugin_moss_image.main import MossImagePlugin  # noqa: E402
from astrbot_plugin_moss_image.prompt_parser import parse_draw_request  # noqa: E402


class FakeEvent:
    def __init__(self) -> None:
        self.unified_msg_origin = "test:FriendMessage:42"
        self.message_str = ""
        self.sent: list = []

    def get_sender_id(self) -> str:
        return "42"

    def is_admin(self) -> bool:
        return True

    def plain_result(self, text: str) -> Plain:
        return Plain(text)

    async def send(self, result) -> None:
        self.sent.append(result)

    def chain_result(self, components):
        # Mirrors AstrMessageEvent.chain_result: the respond stage only sends
        # results that carry a `.chain` list.
        return types.SimpleNamespace(chain=list(components))


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def text_chat(self, prompt=None, system_prompt=None, contexts=None, **kwargs):
        self.calls += 1
        return type("Response", (), {"completion_text": self.reply})()


class FakeContext:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    async def get_using_provider_async(self, umo=None):
        return self.provider

    def get_using_provider(self, umo=None):
        return self.provider


CAPTION = (
    "A lone red fox stands in a snow-covered pine forest at dusk, thick winter fur "
    "rimmed with frost, warm amber eyes reflecting the last light, soft volumetric "
    "mist between the trunks, shallow depth of field, muted blue palette with a warm "
    "accent on the animal, cinematic wildlife photography."
)

LAYOUT_DRAFT = {
    "width": 768,
    "height": 512,
    "bbox_coord_space": "abs",
    "bbox_format": "xyxy",
    "summary": {"caption": "A travel poster for a mountain lake.", "canvas": "single"},
    "items": [
        {"label": "object", "caption": "A still alpine lake reflecting snowy peaks.", "bbox": [40, 140, 470, 470]},
        {"label": "text", "caption": "Large bold white title.", "text": "静 山", "bbox": [500, 80, 730, 200]},
    ],
}


def build_plugin(provider: FakeProvider) -> MossImagePlugin:
    plugin = MossImagePlugin(context=FakeContext(provider), config={})
    plugin.settings.base_url = SERVER
    plugin.settings.width, plugin.settings.height = 512, 512
    plugin.settings.steps = 4
    plugin.client.base_url = SERVER
    plugin.client.generation_timeout = 300
    return plugin


def _texts(components) -> list[str]:
    return [part.text for part in components if isinstance(part, Plain)]


async def run() -> int:
    event = FakeEvent()

    # 1. Plain draw with --raw (no LLM involved).
    plugin = build_plugin(FakeProvider(CAPTION))
    request = parse_draw_request("a red fox in snow --size 512x512 --steps 4 --seed 11 --raw")
    components = await plugin._render(event, request)
    assert isinstance(components[0], Plain), components
    assert "绘制完成" in components[0].text, components
    images = [part for part in components if getattr(part, "file", "").startswith("base64://")]
    assert images, "no image component in the chain"
    assert any("seed 11" in text for text in _texts(components))
    print(f"  draw(--raw)   : {len(components)} components, image {len(images[0].file)} chars")

    # 2. The command path must yield a result object the respond stage sends.
    class CommandEvent(FakeEvent):
        def __init__(self) -> None:
            super().__init__()
            self.message_str = "/draw a red fox --size 512x512 --steps 4 --seed 11 --raw"

    command_event = CommandEvent()
    yielded = [item async for item in plugin.draw(command_event)]
    assert yielded, "draw yielded nothing"
    last = yielded[-1]
    assert hasattr(last, "chain"), f"last yield is {type(last).__name__}, not a MessageEventResult"
    assert any(getattr(part, "file", "").startswith("base64://") for part in last.chain), "no image in result"
    print(f"  /draw command : {len(yielded)} yields, last carries {len(last.chain)} components")

    # 3. Prompt Enhancer runs and its caption is what the server receives.
    provider = FakeProvider(CAPTION)
    plugin = build_plugin(provider)
    request = parse_draw_request("a red fox --size 512x512 --steps 4 --seed 12 --enhance")
    components = await plugin._render(event, request)
    assert provider.calls == 1, provider.calls
    assert any("已增强提示词" in text for text in _texts(components))
    print(f"  draw(--enhance): llm calls={provider.calls}")

    # 4. Layout Enhancer: draft -> compile -> generate.
    provider = FakeProvider(json.dumps(LAYOUT_DRAFT, ensure_ascii=False))
    plugin = build_plugin(provider)
    request = parse_draw_request("a mountain lake poster --size 768x512 --steps 4 --layout")
    components = await plugin._render(event, request)
    assert provider.calls == 1, provider.calls
    assert any("Layout" in text for text in _texts(components))
    print(f"  draw(--layout) : llm calls={provider.calls}, image ok")

    # 5. Layout retry: first reply is broken JSON, second is valid.
    class RetryProvider(FakeProvider):
        def __init__(self, replies):
            super().__init__(replies[0])
            self.replies = list(replies)

        async def text_chat(self, prompt=None, system_prompt=None, contexts=None, **kwargs):
            self.calls += 1
            return type("Response", (), {"completion_text": self.replies.pop(0)})()

    provider = RetryProvider(["not json at all", json.dumps(LAYOUT_DRAFT)])
    plugin = build_plugin(provider)
    result = await plugin.layout.build("a mountain lake poster", 768, 512, umo=event.unified_msg_origin)
    assert provider.calls == 2, provider.calls
    compiled = json.loads(result.prompt)
    assert compiled["bbox_coord_space"] == "abs_centralized"
    print(f"  layout retry   : recovered after {provider.calls} attempts")

    # 6. Status command talks to the live server.
    plugin = build_plugin(FakeProvider(CAPTION))
    status = await plugin.client.status()
    assert status.get("state") in ("idle", "ready", "loading"), status.get("state")
    loras = ", ".join(item["name"] for item in (status.get("loras") or [])) or "none"
    print(f"  server status  : {status.get('state')}, engine={status.get('engine')}, loras={loras}")

    # 7. The LLM sees the LoRA catalog (names + descriptions + weights).
    plugin = build_plugin(FakeProvider(CAPTION))
    for entry in plugin.catalog.entries:
        entry.registered = True

    class FakeLLMRequest:
        system_prompt = ""
        func_tool = types.SimpleNamespace(names=lambda: ["moss_draw_image"])

    llm_request = FakeLLMRequest()
    await plugin._inject_tool_guidance(event, llm_request)
    prompt = llm_request.system_prompt
    assert "【可用 LoRA】" in prompt, prompt[:200]
    assert "kemono" in prompt and "xuenian" in prompt
    assert "兽人" in prompt, "catalog description missing from the system prompt"
    print(f"  llm guidance   : catalog brief injected ({len(prompt)} chars)")

    # 8. The tool's lora argument is parsed against the catalog.
    assert plugin._parse_lora_argument("xuenian:0.8,kemono") == {"xuenian": 0.8, "kemono": 0.7}
    print("  lora argument  : parsed with catalog defaults")

    # 9. Sampling settings are editable (negative prompt / shift / CFG skip).
    from astrbot_plugin_moss_image.main import _parse_skip_layers, parse_settings

    tuned = parse_settings(
        {
            "generation": {
                "steps": 24,
                "guidance": 2.5,
                "negative_prompt": " low quality, blurry ",
                "shift": 4.0,
                "uncond_skip_layers": "22, 24",
            }
        }
    )
    assert tuned.steps == 24 and tuned.guidance == 2.5
    assert tuned.negative_prompt == "low quality, blurry"
    assert tuned.shift == 4.0 and tuned.uncond_skip_layers == (22, 24)
    assert _parse_skip_layers("") is None
    assert _parse_skip_layers("off") == ()
    assert parse_settings({"generation": {"shift": 0}}).shift is None
    print("  sampler config : steps/guidance/negative/shift/skip parsed")

    # 10. The server honours per-request shift + CFG skip overrides.
    tuned_plugin = build_plugin(FakeProvider(CAPTION))
    tuned_plugin.settings.shift = 4.0
    tuned_plugin.settings.uncond_skip_layers = ()
    tuned_outcome = await tuned_plugin.client.generate(
        prompt="a red fox", width=512, height=512, steps=4, guidance=3.0,
        shift=tuned_plugin.settings.shift,
        uncond_skip_layers=tuned_plugin.settings.uncond_skip_layers,
    )
    assert tuned_outcome.images, "no image returned with sampler overrides"
    print(f"  sampler live   : shift=4.0 skip=off -> {len(tuned_outcome.images)} image(s)")

    print("plugin integration: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
