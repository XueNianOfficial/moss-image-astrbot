"""Tests for the prompt-skill loader and the skill output contract.

Runs without AstrBot or a server: only the pure helpers are imported.  The real
skill file is optional - set ``MOSS_TEST_SKILL`` to a ``SKILL.md`` to exercise
section slicing against it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(os.environ.get("ASTRBOT_PLUGINS_DIR") or (Path(__file__).resolve().parents[1] / "plugin"))
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from astrbot_plugin_moss_image.prompt_skill import (  # noqa: E402
    PromptSkill,
    SkillPrompter,
    clean_caption,
    mode_instruction,
)


SAMPLE_SKILL = """---
name: demo-skill
version: 9.9.9
license: CC-BY-NC-ND-4.0
description: >-
  demo
---

# Demo skill

## 0. 任务定义

写提示词。

## 1. 词条清洗

清洗 danbooru tag。

## 2. 角色

角色描述。

## 10. 模板

正负模板。
"""


def write_skill(directory: Path, text: str = SAMPLE_SKILL) -> Path:
    path = directory / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_loader_reads_frontmatter_and_sections(tmp_path: Path) -> None:
    path = write_skill(tmp_path)
    skill = PromptSkill(str(path), sections="0,2")
    info = skill.info()
    assert skill.available
    assert info.name == "demo-skill"
    assert info.version == "9.9.9"
    assert info.license == "CC-BY-NC-ND-4.0"
    assert info.sections == [0, 1, 2, 10], info.sections
    excerpt = skill.excerpt()
    assert "任务定义" in excerpt and "角色描述" in excerpt
    assert "词条清洗" not in excerpt, "section 1 was not requested"
    print(f"  sections sliced: {len(excerpt)} chars")


def test_loader_defaults_and_missing(tmp_path: Path) -> None:
    # A directory is accepted as well as a file.
    write_skill(tmp_path)
    assert PromptSkill(str(tmp_path)).available
    # Nothing configured and no default location on this machine -> unavailable.
    from astrbot_plugin_moss_image import prompt_skill

    saved = prompt_skill.DEFAULT_SKILL_PATHS
    prompt_skill.DEFAULT_SKILL_PATHS = ()
    try:
        missing = PromptSkill(str(tmp_path / "nope"))
        assert missing.path is None
        assert missing.excerpt() == ""
        assert missing.guide() == ""
    finally:
        prompt_skill.DEFAULT_SKILL_PATHS = saved
    print("  directory + missing paths handled")


def test_guide_mentions_path(tmp_path: Path) -> None:
    path = write_skill(tmp_path)
    guide = PromptSkill(str(path)).guide()
    assert "必读" in guide and str(path) in guide
    print("  guide points at the skill file")


def test_max_chars_truncates(tmp_path: Path) -> None:
    path = write_skill(tmp_path, SAMPLE_SKILL * 20)
    skill = PromptSkill(str(path), sections="", max_chars=200)
    excerpt = skill.excerpt()
    assert len(excerpt) <= 260, len(excerpt)
    assert excerpt.endswith("（后文省略）")
    print("  max_chars truncates with a marker")


def test_prompter_parses_json_contract(tmp_path: Path) -> None:
    path = write_skill(tmp_path)
    prompter = SkillPrompter(None, PromptSkill(str(path)))
    system = prompter.system_prompt()
    assert "demo-skill" in system and '"positive"' in system

    result = prompter.parse(
        '```json\n{"positive": "a red fox in snow", "negative": "blurry", '
        '"params": {"steps": 16, "guidance_scale": 3.0, "size": "1024x1024"}}\n```'
    )
    assert result.used_llm
    assert result.positive == "a red fox in snow"
    assert result.negative == "blurry"
    assert result.params["steps"] == 16
    print("  JSON contract parsed")

    # A plain caption is still usable.
    plain = prompter.parse("A lone fox stands in a snowy pine forest at dusk, warm light.")
    assert plain.used_llm and plain.positive.startswith("A lone fox")
    print("  prose fallback parsed")


def test_clean_caption_strips_labels() -> None:
    assert clean_caption("**Prompt:** a cat") == "a cat"
    assert clean_caption('"a cat"') == "a cat"
    assert clean_caption("提示词：一只猫") == "一只猫"
    assert mode_instruction("anime").count("二次元") == 1


def test_real_skill_if_available() -> None:
    path = os.environ.get("MOSS_TEST_SKILL")
    if not path or not Path(path).is_file():
        print("  skipped: set MOSS_TEST_SKILL to a real SKILL.md")
        return
    skill = PromptSkill(path)
    info = skill.info()
    assert info.available and info.chars > 1000
    assert 0 in info.sections
    excerpt = skill.excerpt()
    assert excerpt and len(excerpt) < info.chars, "default sections should be a subset"
    print(f"  real skill: {info.name} v{info.version} ({info.license}), {info.chars} chars -> excerpt {len(excerpt)}")


def _run() -> int:
    import tempfile

    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        print(f"[{test.__name__}]")
        try:
            if test.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print("  ok")
        except AssertionError as error:
            failures += 1
            print(f"  FAILED: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
