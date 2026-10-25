"""Parse ``/draw`` command text into a request the MOSS-Image server understands.

Example::

    /draw 一只戴墨镜的橘猫 --size 1024x1024 --steps 30 --seed 7 \
          --no blurry, watermark --lora kemono:0.8 --layout
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ``--flag value`` / ``--flag=value``
_FLAG_RE = re.compile(
    r"""(?:^|\s)--(?P<key>[a-zA-Z][\w-]*)
        (?:[\s=]+(?P<value>"[^"]*"|'[^']*'|\S+))?
    """,
    re.VERBOSE,
)

_SIZE_RE = re.compile(r"^\s*(\d{2,5})\s*[x*×:]\s*(\d{2,5})\s*$")
_RATIO_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*[:：/]\s*(\d{1,3}(?:\.\d+)?)\s*$")

# ``--no ...`` / ``--negative ...`` swallow everything up to the next flag, so
# commas and spaces inside the negative prompt are preserved.
_NEGATIVE_RE = re.compile(
    r"--(?:no|neg|negative)\s+(?P<text>.+?)(?=\s+--[a-zA-Z]|$)", re.IGNORECASE | re.DOTALL
)

MIN_SIDE, MAX_SIDE, STEP = 64, 2048, 16
DEFAULT_AREA = 1024 * 1024


@dataclass
class DrawRequest:
    text: str = ""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    n: int = 1
    negative: str = ""
    loras: dict[str, float] = field(default_factory=dict)
    #: "auto" (decide by content), "caption" (plain caption), "layout", "raw"
    mode: str = "auto"
    warnings: list[str] = field(default_factory=list)

    @property
    def wants_layout(self) -> bool:
        return self.mode == "layout"

    @property
    def wants_enhance(self) -> bool:
        return self.mode in ("auto", "caption", "layout")


def _snap(value: float) -> int:
    return int(max(MIN_SIDE, min(MAX_SIDE, round(value / STEP) * STEP)))


def size_from_ratio(text: str) -> tuple[int, int] | None:
    match = _RATIO_RE.match(text or "")
    if not match:
        return None
    try:
        ratio_w, ratio_h = float(match.group(1)), float(match.group(2))
    except ValueError:
        return None
    if ratio_w <= 0 or ratio_h <= 0:
        return None
    ratio = ratio_w / ratio_h
    width = math.sqrt(DEFAULT_AREA * ratio)
    height = width / ratio
    return _snap(width), _snap(height)


def parse_size(text: str) -> tuple[int, int] | None:
    match = _SIZE_RE.match(text or "")
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        return None
    if width % STEP or height % STEP:
        # Snap to the nearest legal multiple instead of rejecting outright.
        width, height = _snap(width), _snap(height)
    return width, height


def _as_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value, default=None):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_draw_request(raw: str) -> DrawRequest:
    """Split flags off the prompt text and validate them."""
    request = DrawRequest()
    text = (raw or "").strip()
    flags: dict[str, str] = {}

    negative = ""
    negative_match = _NEGATIVE_RE.search(text)
    if negative_match:
        negative = negative_match.group("text").strip().strip(",， ")
        text = f"{text[: negative_match.start()]} {text[negative_match.end():]}"

    def _strip(match: re.Match) -> str:
        key = match.group("key").lower()
        value = match.group("value")
        if value is not None and len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        normalised = {
            "ratio": "ar", "aspect": "ar", "w": "width", "h": "height",
            "cfg": "guidance", "scale": "guidance", "num": "n", "batch": "n",
            "negative": "no", "neg": "no",
        }.get(key, key)
        flags[normalised] = value if value is not None else "true"
        return " "

    text = _FLAG_RE.sub(_strip, text)
    text = re.sub(r"\s+", " ", text).strip(" ,，" )
    request.text = text

    if not text:
        request.warnings.append("没有画面描述")

    size_value = flags.get("size") or flags.get("ar")
    if size_value:
        parsed = parse_size(size_value) or size_from_ratio(size_value)
        if parsed:
            request.width, request.height = parsed
        else:
            request.warnings.append(f"无法识别的尺寸：{size_value}")

    if flags.get("width"):
        width = _as_int(flags["width"])
        if width:
            request.width = _snap(width)
    if flags.get("height"):
        height = _as_int(flags["height"])
        if height:
            request.height = _snap(height)

    if flags.get("steps"):
        steps = _as_int(flags["steps"])
        if steps and 1 <= steps <= 200:
            request.steps = steps
        else:
            request.warnings.append("步数需在 1-200 之间")

    if flags.get("guidance"):
        guidance = _as_float(flags["guidance"])
        if guidance is not None and 0 <= guidance <= 20:
            request.guidance = guidance
        else:
            request.warnings.append("引导强度需在 0-20 之间")

    if flags.get("seed"):
        seed = _as_int(flags["seed"])
        if seed is not None and seed >= 0:
            request.seed = seed
        else:
            request.warnings.append("种子需为非负整数")

    if flags.get("n"):
        count = _as_int(flags["n"])
        if count and 1 <= count <= 4:
            request.n = count
        else:
            request.warnings.append("数量需在 1-4 之间")

    if negative:
        request.negative = negative
    elif flags.get("no"):
        request.negative = str(flags["no"]).strip()

    if flags.get("lora"):
        for item in re.split(r"[,，;；]", str(flags["lora"])):
            item = item.strip()
            if not item:
                continue
            name, _, weight = item.partition(":")
            try:
                strength = float(weight) if weight else 1.0
            except ValueError:
                strength = 1.0
            if name.strip():
                request.loras[name.strip()] = strength

    if "layout" in flags or "布局" in flags:
        request.mode = "layout"
    elif "enhance" in flags or "增强" in flags:
        request.mode = "caption"
    elif "raw" in flags or "原样" in flags:
        request.mode = "raw"

    return request
