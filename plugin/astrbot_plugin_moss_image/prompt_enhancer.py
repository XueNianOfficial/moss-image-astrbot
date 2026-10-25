"""Prompt Enhancer: turn a short idea into a detailed MOSS-Image caption.

The rules are distilled from the official ``moss-image-prompt-enhancer`` skill
(``G:\\MOSS-Image\\skills``): develop visible choices, preserve the user's
subjects/counts/visible text, and never pad the prompt with generic quality
tags.  The LLM does the writing; this module only builds the instructions and
cleans up the reply.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

SYSTEM_PROMPT = """You are a prompt writer for MOSS-Image, a text-to-image diffusion model.

Turn the user's idea into ONE cohesive image caption that the model can render.

Rules:
- Describe only visible content: subject and action, relevant appearance, setting,
  camera/framing, medium, lighting, materials, palette, mood.
- Preserve the user's subjects, their counts, relationships, style, medium and
  exclusions. Resolve ordinary creative gaps yourself; never invent facts.
- If the user quotes text that should appear in the image, keep it verbatim
  (same language, punctuation and line breaks) and describe how it is rendered.
- Write flowing prose, not a tag list. No "masterpiece, best quality, 8k" padding,
  no artist names, no meta commentary, no instructions about JSON or tools.
- Be specific and visual; 60-140 words is usually right.

Output only the caption itself: no preamble, no quotes, no markdown, no labels."""

_LABEL_RE = re.compile(
    r"^\s*(?:caption|prompt|final prompt|提示词|画面描述|描述)\s*[:：]\s*", re.IGNORECASE
)


@dataclass
class Enhancement:
    text: str
    used_llm: bool = False
    warning: str = ""


class PromptEnhancer:
    def __init__(self, context, *, enabled: bool = True, max_chars: int = 1400, timeout: float = 90.0):
        self.context = context
        self.enabled = enabled
        self.max_chars = max(200, int(max_chars or 1400))
        self.timeout = timeout

    async def enhance(self, idea: str, *, umo: str | None = None, hint: str = "") -> Enhancement:
        idea = (idea or "").strip()
        if not idea:
            return Enhancement(text="", warning="没有可增强的描述")
        if not self.enabled:
            return Enhancement(text=idea)

        provider = await self._provider(umo)
        if provider is None:
            return Enhancement(text=idea, warning="没有可用的 LLM 供应商，已跳过提示词增强")

        user_prompt = idea
        if hint:
            user_prompt = f"{idea}\n\nAdditional direction: {hint}"
        try:
            response = await asyncio.wait_for(
                provider.text_chat(
                    prompt=user_prompt, system_prompt=SYSTEM_PROMPT, contexts=[]
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return Enhancement(text=idea, warning="提示词增强超时，已使用原始描述")
        except Exception as error:  # noqa: BLE001 - never break drawing because of the LLM
            return Enhancement(text=idea, warning=f"提示词增强失败（{type(error).__name__}），已使用原始描述")

        text = clean_caption(getattr(response, "completion_text", "") or "")
        if not text:
            return Enhancement(text=idea, warning="提示词增强返回了空内容，已使用原始描述")
        text = self._truncate(text)
        if len(text) < 20:
            return Enhancement(text=idea, warning="提示词增强结果过短，已使用原始描述")
        return Enhancement(text=text, used_llm=True)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        cut = text[: self.max_chars]
        end = max(cut.rfind(". "), cut.rfind("。"), cut.rfind("\n"))
        return (cut[: end + 1] if end > 40 else cut).strip()

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


def clean_caption(text: str) -> str:
    """Strip markdown fences, labels and wrapping quotes from an LLM reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = _LABEL_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'“”「」" and text[-1] in "\"'“”「」":
        text = text[1:-1].strip()
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
