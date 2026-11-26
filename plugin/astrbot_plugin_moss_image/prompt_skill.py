"""Prompt skill loader (moss-image-prompt-skill).

The plugin intentionally carries **no** prompt-writing rules of its own.  The
rules live in https://github.com/Moeblack/moss-image-prompt-skill (CC BY-NC-ND
4.0) and are read from disk at runtime, so the skill can be updated - or swapped
for a fork - without touching this plugin.  Nothing from that repository is
vendored here, which also keeps the plugin MIT-compatible.

Two ways the skill is used:

* ``guide()`` - a short instruction injected into the system prompt telling the
  model to read the skill file before it writes a prompt (the agent-side path;
  the model reads the whole file itself with its file tools).
* ``SkillPrompter.rewrite()`` - one LLM call with a skill excerpt as the system
  prompt, used by ``/draw --enhance`` and by callers that want a ready-made
  positive/negative pair without an agent loop.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .layout_enhancer import extract_json

#: Where AstrBot keeps agent skills, plus the raw clone location.
DEFAULT_SKILL_PATHS = (
    "~/.astrbot/data/skills/moss-image-prompt-skill/SKILL.md",
    "~/.astrbot/data/skills/moss-image-prompt-skill/skill.md",
)

SKILL_REPO = "https://github.com/Moeblack/moss-image-prompt-skill"

_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s*(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

#: Sections that cover the writing flow end to end; used when the operator does
#: not configure an explicit subset.  Section 0 = task/defaults, 1 = tag
#: cleaning, 2 = characters, 3 = medium and lighting, 10 = templates/self-check.
DEFAULT_SECTIONS = "0,1,2,3,10"


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


@dataclass
class SkillInfo:
    path: str | None = None
    name: str = "moss-image-prompt-skill"
    version: str = ""
    license: str = ""
    chars: int = 0
    sections: list[int] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.path)


class PromptSkill:
    """Resolve, cache and slice the on-disk skill file."""

    def __init__(self, path: str = "", *, sections: str = DEFAULT_SECTIONS, max_chars: int = 0):
        self.configured_path = (path or "").strip()
        self.sections = [int(part) for part in re.findall(r"\d+", sections or "")]
        self.max_chars = max(0, int(max_chars or 0))
        self._cache: str | None = None
        self._info: SkillInfo | None = None

    # ----------------------------------------------------------- resolution
    def resolve(self) -> Path | None:
        candidates: list[Path] = []
        configured = self.configured_path
        if configured:
            candidates.append(_expand(configured))
        candidates.extend(_expand(item) for item in DEFAULT_SKILL_PATHS)
        for candidate in candidates:
            if candidate.is_dir():
                for name in ("SKILL.md", "skill.md"):
                    if (candidate / name).is_file():
                        return candidate / name
            elif candidate.is_file():
                return candidate
        return None

    @property
    def path(self) -> Path | None:
        return self.resolve()

    @property
    def available(self) -> bool:
        return self.path is not None

    def text(self) -> str:
        if self._cache is not None:
            return self._cache
        path = self.path
        if path is None:
            return ""
        try:
            self._cache = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return self._cache

    def info(self) -> SkillInfo:
        if self._info is not None:
            return self._info
        path = self.path
        text = self.text()
        info = SkillInfo(path=str(path) if path else None, chars=len(text))
        if text:
            front = _FRONTMATTER_RE.match(text)
            if front:
                for line in front.group(1).splitlines():
                    key, _, value = line.partition(":")
                    key, value = key.strip().lower(), value.strip()
                    if key == "name" and value:
                        info.name = value
                    elif key == "version" and value:
                        info.version = value
                    elif key == "license" and value:
                        info.license = value
            info.sections = [int(number) for number, _ in _SECTION_RE.findall(text)]
        self._info = info
        return info

    # -------------------------------------------------------------- excerpt
    def excerpt(self) -> str:
        """Whole skill text, or only the configured sections."""
        text = self.text()
        if not text:
            return ""
        if self.sections:
            text = self._slice(text, self.sections)
        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n\n（后文省略）"
        return text

    @staticmethod
    def _slice(text: str, wanted: list[int]) -> str:
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            return text
        chunks: list[str] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            if int(match.group(1)) in wanted:
                chunks.append(text[start:end].rstrip())
        return "\n\n".join(chunks)

    # ---------------------------------------------------------------- guide
    def guide(self) -> str:
        """Instruction telling the model to consult the skill before drawing."""
        path = self.path
        if path is None:
            return ""
        return (
            "【提示词规范（必读）】只要本轮要调用绘图/改图工具，必须先读取提示词规范文件：\n"
            f"{path}\n"
            "先看它的 §0 与目录判断模式，再只读命中的章节，不要整篇灌入；"
            "按其中的「模式判断 → booru tag 清洗 → 分块语序 → 正负模板 → 自检清单」流程写好提示词，"
            "再带着提示词调用工具。禁止跳过该文件凭直觉写提示词；确实无法读取时须简短说明原因。"
        )


@dataclass
class SkillRewrite:
    positive: str
    negative: str = ""
    params: dict = field(default_factory=dict)
    used_llm: bool = False
    warning: str = ""


OUTPUT_CONTRACT = """\
Output exactly one JSON object, no markdown fence, no commentary:
{"positive": "<the final positive prompt>", "negative": "<negative prompt>", "params": {"steps": 16, "guidance_scale": 3.0, "size": "1024x1024"}}
Rules for the JSON:
- `positive` must follow the skill rules above and stay in the language that fits the skill
  (the skill's own examples decide; usually English prose).
- `negative` is the negative prompt the skill's template asks for (empty string if none).
- `params` carries only what the skill recommends for this request; omit keys you do not need.
  Allowed keys: steps (int), guidance_scale (number), size ("WxH"), seed (int)."""


class SkillPrompter:
    """One LLM call that turns an idea into a skill-compliant prompt pair."""

    def __init__(self, context, skill: PromptSkill, *, enabled: bool = True, timeout: float = 150.0):
        self.context = context
        self.skill = skill
        self.enabled = enabled
        self.timeout = timeout

    def system_prompt(self) -> str:
        excerpt = self.skill.excerpt()
        if not excerpt:
            return ""
        header = (
            f"You are compiling a prompt for MOSS-Image using the skill "
            f"`{self.skill.info().name}`. Follow the skill rules below as the single source of truth.\n\n"
        )
        return f"{header}{excerpt}\n\n{OUTPUT_CONTRACT}"

    async def rewrite(
        self, idea: str, *, umo: str | None = None, mode: str = "auto", hint: str = ""
    ) -> SkillRewrite:
        idea = (idea or "").strip()
        if not idea:
            return SkillRewrite(positive="", warning="没有可增强的描述")
        if not self.enabled:
            return SkillRewrite(positive=idea)
        system_prompt = self.system_prompt()
        if not system_prompt:
            return SkillRewrite(
                positive=idea,
                warning=(
                    "未找到提示词 skill 文件（moss-image-prompt-skill），已使用原始描述；"
                    f"可从 {SKILL_REPO} 获取并在配置里填写路径"
                ),
            )
        provider = await self._provider(umo)
        if provider is None:
            return SkillRewrite(positive=idea, warning="没有可用的 LLM 供应商，已跳过提示词增强")

        prompt = f"{mode_instruction(mode)}\n\n用户需求：{idea}"
        if hint:
            prompt = f"{prompt}\n\n补充方向：{hint}"
        try:
            response = await asyncio.wait_for(
                provider.text_chat(prompt=prompt, system_prompt=system_prompt, contexts=[]),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return SkillRewrite(positive=idea, warning="提示词增强超时，已使用原始描述")
        except Exception as error:  # noqa: BLE001 - never block drawing
            return SkillRewrite(
                positive=idea, warning=f"提示词增强失败（{type(error).__name__}），已使用原始描述"
            )

        reply = getattr(response, "completion_text", "") or ""
        return self.parse(reply, fallback=idea)

    def parse(self, reply: str, *, fallback: str = "") -> SkillRewrite:
        reply = (reply or "").strip()
        if not reply:
            return SkillRewrite(positive=fallback, warning="提示词增强返回了空内容，已使用原始描述")
        try:
            data = extract_json(reply)
        except ValueError:
            text = clean_caption(reply)
            if len(text) < 20:
                return SkillRewrite(positive=fallback, warning="提示词增强结果无法解析，已使用原始描述")
            return SkillRewrite(positive=text, used_llm=True)
        positive = clean_caption(str(data.get("positive") or data.get("prompt") or ""))
        negative = clean_caption(str(data.get("negative") or data.get("negative_prompt") or ""))
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        if not positive:
            return SkillRewrite(positive=fallback, warning="提示词增强没有给出正面提示词，已使用原始描述")
        return SkillRewrite(positive=positive, negative=negative, params=params, used_llm=True)

    async def _provider(self, umo: str | None):
        getter = getattr(self.context, "get_using_provider_async", None)
        try:
            if getter is not None:
                return await getter(umo=umo)
        except Exception:  # noqa: BLE001 - fall back to the sync accessor
            pass
        try:
            return self.context.get_using_provider(umo)
        except Exception:  # noqa: BLE001
            return None


def mode_instruction(mode: str) -> str:
    if mode == "real":
        return "按 skill 的**三次元真人写实**模式（默认模式）编写提示词。"
    if mode == "anime":
        return "按 skill 的**二次元**模式编写提示词（正文保持画风中立）。"
    if mode == "cross":
        return "按 skill 的**跨次元**模式编写提示词（2D 角色置身 3D 场景或相反）。"
    return "先按 skill 的 §0 判断该用写实、二次元还是跨次元模式，再按对应规则编写。"


_LABEL_RE = re.compile(
    r"^\s*(?:caption|prompt|final prompt|positive|提示词|画面描述|描述)\s*[:：]\s*", re.IGNORECASE
)


def clean_caption(text: str) -> str:
    """Strip markdown fences, labels and wrapping quotes from an LLM reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # ``**Prompt:** x`` / ``## 提示词: x`` -> ``x``
    text = re.sub(r"^[\s*_#>\-]+", "", text)
    text = _LABEL_RE.sub("", text).strip()
    text = re.sub(r"^[\s*_#]+", "", text)
    if len(text) >= 2 and text[0] in "\"'“”「」" and text[-1] in "\"'“”「」":
        text = text[1:-1].strip()
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
