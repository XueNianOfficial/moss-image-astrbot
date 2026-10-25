"""Layout Enhancer: LLM drafts pixel boxes, ``compile_prompt`` validates them.

The model-facing prompt is a single line of JSON with centred ``yxyx`` boxes.
The LLM only ever writes the *draft* (absolute pixel ``xyxy``); coordinate
arithmetic and validation stay in the vendored official compiler, exactly as
the upstream ``moss-image-prompt-enhancer`` skill prescribes.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from .compile_prompt import compile_prompt

SYSTEM_PROMPT = """You are a layout planner for MOSS-Image, a diffusion model that can place
text and objects at exact positions.

Return ONLY one JSON object (no markdown, no prose) with this exact schema:
{
  "width": <canvas width in px>,
  "height": <canvas height in px>,
  "bbox_coord_space": "abs",
  "bbox_format": "xyxy",
  "summary": {"caption": "<global description of the whole image>",
              "meta": ["<medium/genre tag>"], "canvas": "single"},
  "items": [
    {"label": "character" | "object" | "text",
     "caption": "<what this element looks like, and for text how it is set>",
     "bbox": [x1, y1, x2, y2],
     "text": "<exact visible characters, only for label=text>",
     "tags": ["optional"], "palette": ["optional"]}
  ]
}

Hard rules:
- Use exactly the canvas size given by the user. Never change width/height.
- bbox is [x_min, y_min, x_max, y_max] in absolute pixels from the top-left;
  x1 < x2, y1 < y2, and every value inside the canvas.
- items must be a non-empty flat array. No nesting, no ids, no extra fields.
- label "text" items MUST carry "text" with the exact visible string; use
  separate items when placement differs. Keep the user's wording verbatim.
- Letter boxes must be large enough to hold the copy at this resolution.
- Describe appearance in captions, never draw borders or panels.
- Output valid JSON only: double quotes, no trailing commas, no comments."""

_JSON_START = re.compile(r"[{]")


class LayoutError(Exception):
    """Raised when no valid draft could be produced."""


@dataclass
class LayoutResult:
    prompt: str
    width: int
    height: int
    draft: dict = field(default_factory=dict)
    attempts: int = 1


class LayoutEnhancer:
    def __init__(
        self,
        context,
        *,
        enabled: bool = True,
        max_retries: int = 2,
        timeout: float = 150.0,
    ):
        self.context = context
        self.enabled = enabled
        self.max_retries = max(0, int(max_retries))
        self.timeout = timeout

    async def build(
        self,
        idea: str,
        width: int,
        height: int,
        *,
        umo: str | None = None,
    ) -> LayoutResult:
        if not self.enabled:
            raise LayoutError("布局增强已在插件配置中关闭")
        provider = await self._provider(umo)
        if provider is None:
            raise LayoutError("没有可用的 LLM 供应商")

        idea = (idea or "").strip()
        if not idea:
            raise LayoutError("没有可排版的画面描述")

        feedback = ""
        last_error = "未知错误"
        for attempt in range(1, self.max_retries + 2):
            user_prompt = (
                f"Canvas: {width}x{height} pixels.\n"
                f"Image idea:\n{idea}\n\n"
                f"Return the layout JSON only.{feedback}"
            )
            try:
                response = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=user_prompt, system_prompt=SYSTEM_PROMPT, contexts=[]
                    ),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as error:
                last_error = "LLM 超时"
                feedback = "\n\n上一条请求超时，请更简洁地输出 JSON。"
                if attempt > self.max_retries:
                    raise LayoutError(last_error) from error
                continue
            except Exception as error:  # noqa: BLE001 - propagate as LayoutError
                raise LayoutError(f"LLM 调用失败（{type(error).__name__}）") from error

            raw = getattr(response, "completion_text", "") or ""
            try:
                draft = extract_json(raw)
            except ValueError as error:
                last_error = f"返回内容不是合法 JSON：{error}"
                feedback = "\n\n你上一条回复不是合法 JSON。只输出一个 JSON 对象。"
                continue

            if not isinstance(draft, dict):
                last_error = "JSON 顶层不是对象"
                feedback = "\n\n顶层必须是 JSON 对象。"
                continue

            # The canvas is authoritative: the caller renders exactly these pixels.
            draft["width"] = int(width)
            draft["height"] = int(height)
            try:
                compiled = compile_prompt(draft)
            except ValueError as error:
                last_error = str(error)
                feedback = (
                    f"\n\n上一条 JSON 校验失败：{error}。"
                    "请修正 bbox 使其位于画布内且 x1<x2、y1<y2，然后只输出修正后的 JSON。"
                )
                continue

            return LayoutResult(prompt=compiled, width=width, height=height, draft=draft, attempts=attempt)

        raise LayoutError(last_error)

    async def _provider(self, umo: str | None):
        getter = getattr(self.context, "get_using_provider_async", None)
        try:
            if getter is not None:
                return await getter(umo=umo)
        except Exception:  # noqa: BLE001 - fall back to sync accessor
            pass
        try:
            return self.context.get_using_provider(umo)
        except Exception:  # noqa: BLE001
            return None


def extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of an LLM reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_START.search(text)
    if not match:
        raise ValueError("没有找到 JSON 对象")
    start = match.start()
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as error:
                    raise ValueError(str(error)) from error
    raise ValueError("JSON 不完整（缺少右花括号）")
