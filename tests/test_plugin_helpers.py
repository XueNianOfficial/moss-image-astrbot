"""Checks for the AstrBot plugin helpers (no AstrBot runtime required).

The plugin lives in AstrBot's data directory; the pure-python helpers
(``prompt_parser``, ``prompt_enhancer.clean_caption``, ``layout_enhancer``)
are imported from there directly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(
    os.environ.get("ASTRBOT_PLUGINS_DIR")
    or (Path(__file__).resolve().parents[1] / "plugin")
)
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from astrbot_plugin_moss_image.layout_enhancer import extract_json  # noqa: E402
from astrbot_plugin_moss_image.prompt_enhancer import clean_caption  # noqa: E402
from astrbot_plugin_moss_image.prompt_parser import parse_draw_request, size_from_ratio  # noqa: E402


def test_basic_flags():
    request = parse_draw_request("一只白狐 --size 1024x1536 --steps 30 --seed 7 --lora kemono:0.8")
    assert request.text == "一只白狐"
    assert (request.width, request.height) == (1024, 1536)
    assert request.steps == 30
    assert request.seed == 7
    assert request.loras == {"kemono": 0.8}
    assert request.mode == "auto"


def test_negative_prompt_swallows_spaces_and_commas():
    request = parse_draw_request("a fox --no blurry, 水印, extra fingers --steps 12")
    assert request.negative == "blurry, 水印, extra fingers"
    assert request.steps == 12
    assert "水印" not in request.text
    assert request.text == "a fox"


def test_negative_at_end():
    request = parse_draw_request("a fox --no low quality")
    assert request.negative == "low quality"
    assert request.text == "a fox"


def test_ratio_sizes_are_legal():
    for ratio, expected in (("16:9", (1360, 768)), ("9:16", (768, 1360)), ("1:1", (1024, 1024))):
        assert size_from_ratio(ratio) == expected, (ratio, size_from_ratio(ratio))
        width, height = size_from_ratio(ratio)
        assert width % 16 == 0 and height % 16 == 0


def test_modes():
    assert parse_draw_request("x --layout").mode == "layout"
    assert parse_draw_request("x --raw").mode == "raw"
    assert parse_draw_request("x --enhance").mode == "caption"
    assert parse_draw_request("just text").mode == "auto"


def test_invalid_values_warn():
    request = parse_draw_request("x --steps 999 --seed -3 --size 1000x1000")
    assert request.steps is None
    assert request.seed is None
    assert request.warnings


def test_clean_caption():
    fenced = "```text\nCaption: A red apple on a wooden table.\n```"
    assert clean_caption(fenced) == "A red apple on a wooden table."
    assert clean_caption('"A cat."') == "A cat."
    assert clean_caption("提示词：一只猫") == "一只猫"


def test_extract_json_handles_prose_and_nesting():
    payload = {"width": 1024, "summary": {"caption": "x"}, "items": [{"label": "text", "text": "a}b"}]}
    text = "这是布局：\n" + json.dumps(payload, ensure_ascii=False) + "\n希望有帮助"
    assert extract_json(text) == payload


def test_compiler_accepts_a_draft():
    from astrbot_plugin_moss_image.layout_enhancer import compile_prompt

    draft = {
        "width": 1024,
        "height": 1024,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": {"caption": "A poster with a title.", "canvas": "single"},
        "items": [
            {"label": "object", "caption": "A cup", "bbox": [100, 200, 400, 700]},
            {"label": "text", "caption": "Title", "text": "夏日限定", "bbox": [500, 120, 900, 260]},
        ],
    }
    compiled = compile_prompt(draft)
    data = json.loads(compiled)
    assert data["bbox_coord_space"] == "abs_centralized"
    assert data["bbox_format"] == "yxyx"
    # yxyx, centred: y1-H/2, x1-W/2, y2-H/2, x2-W/2
    assert data["items"][0]["bbox"] == [-312, -412, 188, -112]
    assert "\n" not in compiled


def test_compiler_rejects_out_of_bounds():
    from astrbot_plugin_moss_image.layout_enhancer import compile_prompt

    draft = {
        "width": 512,
        "height": 512,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": {"caption": "x"},
        "items": [{"label": "object", "caption": "x", "bbox": [0, 0, 900, 900]}],
    }
    try:
        compile_prompt(draft)
    except ValueError as error:
        assert "canvas" in str(error)
        return
    raise AssertionError("out-of-bounds box should be rejected")


def test_lora_catalog_parsing_and_resolution():
    from astrbot_plugin_moss_image.lora_config import LoRACatalog, parse_catalog

    raw = [
        {
            "name": "kemono",
            "path": "G:/x/kemono.safetensors",
            "description": "兽人画风",
            "enabled": True,
            "weight": 0.7,
        },
        {
            "name": "xuenian",
            "path": "G:/x/xuenian.safetensors",
            "description": "角色雪年",
            "enabled": False,
            "weight": 1.0,
            "trigger_words": "xuenian, 雪年",
        },
    ]
    entries = parse_catalog(raw)
    assert [e.name for e in entries] == ["kemono", "xuenian"]
    assert entries[0].enabled is True and entries[0].weight == 0.7
    assert entries[1].trigger_words == "xuenian, 雪年"

    # a dict keyed by name and a plain path list are accepted too
    keyed = parse_catalog({"a": {"path": "G:/a.safetensors", "weight": 0.5}})
    assert keyed[0].name == "a" and keyed[0].weight == 0.5
    assert [e.name for e in parse_catalog(["G:/b.safetensors"])] == ["b"]

    catalog = LoRACatalog(entries, max_weight=1.0)
    for entry in catalog.entries:
        entry.registered = True

    weights, warnings = catalog.resolve()
    assert weights == {"kemono": 0.7, "xuenian": 0.0}, weights
    assert not warnings

    # disabled entry can be enabled per request; weights are clamped
    weights, _ = catalog.resolve({"xuenian": 1.6})
    assert weights["xuenian"] == 1.0, weights
    weights, _ = catalog.resolve({"kemono": 0.3})
    assert weights == {"kemono": 0.3, "xuenian": 0.0}

    weights, warnings = catalog.resolve({"nope": 1.0})
    assert any("未知 LoRA" in w for w in warnings), warnings
    assert weights["kemono"] == 0.7

    assert catalog.trigger_words({"kemono": 0.7, "xuenian": 0.0}) == []
    assert catalog.trigger_words({"kemono": 0.7, "xuenian": 0.9}) == ["xuenian", "雪年"]

    brief = catalog.brief()
    assert "kemono" in brief and "兽人画风" in brief and "默认启用" in brief


def test_lora_catalog_text_format():
    from astrbot_plugin_moss_image.lora_config import parse_catalog

    dsl = (
        "# 注释行会被忽略\n"
        "kemono | G:/x/kemono.safetensors | on | 0.7 | 兽人画风 | \n"
        "xuenian | G:/x/xue.safetensors | off | 1.05 | 角色雪年 | 雪年, xuenian"
    )
    entries = parse_catalog(dsl)
    assert [e.name for e in entries] == ["kemono", "xuenian"], entries
    assert entries[0].enabled is True and entries[0].weight == 0.7
    assert entries[1].enabled is False and entries[1].weight == 1.05
    assert entries[1].description == "角色雪年"
    assert entries[1].trigger_words == "雪年, xuenian"

    # semicolons separate entries on a single line
    one_line = "a | G:/a.safetensors | on | 0.5 | x | ; b | G:/b.safetensors | off | 0.9 | y |"
    assert [e.name for e in parse_catalog(one_line)] == ["a", "b"]

    # a bare path still works, and malformed lines are skipped
    assert [e.name for e in parse_catalog("G:/c.safetensors")] == ["c"]
    assert parse_catalog("only-one-field") == []


def _run() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        print(f"[{test.__name__}]")
        try:
            test()
            print("  ok")
        except AssertionError as error:
            failures += 1
            print(f"  FAILED: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
