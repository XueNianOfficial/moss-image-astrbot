"""Request/response models for the HTTP API."""
from __future__ import annotations

import base64
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .engine import GenerationRequest


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible ``POST /v1/images/generations`` body (with extensions)."""

    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    model: str | None = None
    n: int = Field(1, ge=1, le=8)
    size: str | dict[str, int] | None = None
    width: int | None = None
    height: int | None = None
    response_format: Literal["url", "b64_json"] = "url"
    seed: int | None = None

    steps: int | None = Field(None, ge=1, le=200)
    num_inference_steps: int | None = Field(None, ge=1, le=200)
    guidance_scale: float | None = Field(None, ge=0.0, le=20.0)
    #: Flow-matching sigma shift (vendor default 5.0).
    shift: float | None = Field(None, ge=1.0, le=20.0)
    #: DiT blocks skipped in the unconditional branch; [] disables the skip.
    uncond_skip_layers: list[int] | None = None

    # name -> strength for LoRAs already registered on the server; additional
    # entries are ignored (load them through /v1/loras first).
    loras: dict[str, float] | None = None

    @field_validator("size")
    @classmethod
    def _check_size(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) != {"width", "height"}:
                raise ValueError('size object must contain exactly "width" and "height"')
        return value

    @model_validator(mode="after")
    def _check_steps(self) -> "ImageGenerationRequest":
        if self.steps is not None and self.num_inference_steps is not None:
            if self.steps != self.num_inference_steps:
                raise ValueError("steps and num_inference_steps disagree")
        return self

    def resolve_size(self, default_width: int, default_height: int, max_pixels: int) -> tuple[int, int]:
        width, height = self.width, self.height
        if isinstance(self.size, str):
            parts = self.size.lower().replace("×", "x").split("x")
            if len(parts) != 2:
                raise ValueError('size must look like "1024x1024"')
            width, height = int(parts[0]), int(parts[1])
        elif isinstance(self.size, dict):
            width, height = self.size["width"], self.size["height"]

        width = default_width if width is None else width
        height = default_height if height is None else height
        for label, value in (("width", width), ("height", height)):
            if value % 16:
                raise ValueError(f"{label} must be a multiple of 16, got {value}")
            if not 64 <= value <= 2048:
                raise ValueError(f"{label} must be within 64..2048, got {value}")
        if width * height > max_pixels:
            raise ValueError(
                f"{width}x{height} exceeds the configured max_pixels={max_pixels}"
            )
        return width, height


class NativeGenerationRequest(ImageGenerationRequest):
    """Same body, but unknown fields are rejected instead of ignored."""

    model_config = {"extra": "forbid"}


class LoRALoadRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a .safetensors LoRA file")
    name: str | None = Field(None, description="Registry name; defaults to the file stem")
    scale: float = 1.0
    replace: bool = False


class LoRAScaleRequest(BaseModel):
    scale: float


class MossEditRequest(BaseModel):
    """``POST /v1/moss/edit``: edit reference image(s) with a text instruction."""

    model_config = {"extra": "forbid"}

    prompt: str = Field(..., min_length=1, max_length=8000,
                        description="What to change, e.g. '把背景换成晴天的海边，保留人物'")
    images: list[str] = Field(
        default_factory=list,
        description="1..N reference images as base64 or data: URLs (order matters)",
    )
    negative_prompt: str = ""
    size: str | dict[str, int] | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = Field(None, ge=1, le=200)
    guidance_scale: float | None = Field(None, ge=0.0, le=20.0)
    seed: int | None = None
    shift: float | None = Field(None, ge=1.0, le=20.0)
    max_edge: int | None = Field(None, ge=16, le=2048,
                                 description="Longest edge the reference image is resized to")
    use_kv_cache: bool = True
    response_format: Literal["url", "b64_json"] = "url"

    @field_validator("images")
    @classmethod
    def _check_images(cls, value: list[str]) -> list[str]:
        if not 1 <= len(value) <= 8:
            raise ValueError("provide between 1 and 8 reference images")
        return value


class ImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    seconds: float | None = None


class ImagesResponse(BaseModel):
    created: int
    data: list[ImageData]
    model: str = "moss-image"
    seed: int | None = None
    seconds: float | None = None


def build_generation_request(
    body: ImageGenerationRequest,
    *,
    default_width: int,
    default_height: int,
    default_steps: int,
    default_guidance: float,
    max_pixels: int,
    default_shift: float | None = None,
    default_uncond_skip: list[int] | None = None,
    progress=None,
) -> tuple[GenerationRequest, int, int]:
    width, height = body.resolve_size(default_width, default_height, max_pixels)
    steps = body.steps or body.num_inference_steps or default_steps
    request = GenerationRequest(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=default_guidance if body.guidance_scale is None else body.guidance_scale,
        seed=body.seed,
        num_images=body.n,
        loras=body.loras,
        shift=body.shift if body.shift is not None else default_shift,
        uncond_skip_layers=(
            tuple(body.uncond_skip_layers)
            if body.uncond_skip_layers is not None
            else (tuple(default_uncond_skip) if default_uncond_skip is not None else None)
        ),
        progress=progress,
    )
    return request, width, height


# --------------------------------------------------------------- editing
def decode_edit_images(
    payloads: list[str], *, max_images: int = 4, max_bytes: int = 16 * 1024 * 1024
) -> list[bytes]:
    """Decode base64 / data-URL reference images into raw bytes."""
    if not payloads:
        raise ValueError("at least one reference image is required")
    if len(payloads) > max_images:
        raise ValueError(f"at most {max_images} reference images are supported")
    images: list[bytes] = []
    for index, payload in enumerate(payloads):
        text = (payload or "").strip()
        if not text:
            raise ValueError(f"images[{index}] is empty")
        if text.startswith("data:"):
            comma = text.find(",")
            if comma < 0:
                raise ValueError(f"images[{index}] is not a valid data URL")
            text = text[comma + 1 :]
        try:
            raw = base64.b64decode(text, validate=True)
        except Exception as error:  # noqa: BLE001 - normalised into ValueError
            raise ValueError(f"images[{index}] is not valid base64: {error}") from error
        if not raw:
            raise ValueError(f"images[{index}] decodes to zero bytes")
        if len(raw) > max_bytes:
            raise ValueError(
                f"images[{index}] is {len(raw) / 2**20:.1f} MiB, over the {max_bytes / 2**20:.0f} MiB limit"
            )
        images.append(raw)
    return images


def _align(value: float, limit: int) -> int:
    snapped = int(round(value / 16.0)) * 16
    return max(64, min(limit, snapped or 64))


def _fit_to_reference(raw: bytes, area: int, max_pixels: int) -> tuple[int, int] | None:
    """Canvas with the reference image's aspect ratio at roughly ``area`` pixels."""
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - fall back to the configured default
        return None
    if width <= 0 or height <= 0:
        return None
    scale = math.sqrt(area / float(width * height))
    target_w = max(64.0, width * scale)
    target_h = max(64.0, height * scale)
    limit = 2048
    width_out, height_out = _align(target_w, limit), _align(target_h, limit)
    while width_out * height_out > max_pixels and min(width_out, height_out) > 64:
        width_out = _align(width_out * 0.94, limit)
        height_out = _align(height_out * 0.94, limit)
    return width_out, height_out


def build_edit_request(
    body: "MossEditRequest",
    *,
    default_width: int,
    default_height: int,
    max_pixels: int,
    default_steps: int,
    default_guidance: float,
    default_shift: float,
    max_ref_images: int,
    default_max_edge: int,
    progress=None,
):
    """Validate an edit body and turn it into an ``EditRequest``."""
    from .edit import EditRequest

    images = decode_edit_images(body.images, max_images=max_ref_images)

    width, height = body.width, body.height
    if isinstance(body.size, str):
        parts = body.size.lower().replace("×", "x").split("x")
        if len(parts) != 2:
            raise ValueError('size must look like "1024x1024"')
        width, height = int(parts[0]), int(parts[1])
    elif isinstance(body.size, dict):
        if set(body.size) != {"width", "height"}:
            raise ValueError('size object must contain exactly "width" and "height"')
        width, height = body.size["width"], body.size["height"]

    if width is None and height is None:
        derived = _fit_to_reference(images[0], default_width * default_height, max_pixels)
        if derived is None:
            width, height = default_width, default_height
        else:
            width, height = derived
    else:
        width = default_width if width is None else width
        height = default_height if height is None else height

    for label, value in (("width", width), ("height", height)):
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")
        if not 64 <= value <= 2048:
            raise ValueError(f"{label} must be within 64..2048, got {value}")
    if width * height > max_pixels:
        raise ValueError(f"{width}x{height} exceeds the configured max_pixels={max_pixels}")

    steps = body.steps or default_steps
    request = EditRequest(
        prompt=body.prompt,
        images=images,
        negative_prompt=body.negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=default_guidance if body.guidance_scale is None else body.guidance_scale,
        seed=body.seed,
        shift=body.shift if body.shift is not None else default_shift,
        max_edge=body.max_edge or default_max_edge,
        use_kv_cache=body.use_kv_cache,
        progress=progress,
    )
    return request, width, height
