"""Request/response models for the HTTP API."""
from __future__ import annotations

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
