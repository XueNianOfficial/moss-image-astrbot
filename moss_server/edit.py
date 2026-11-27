"""Reference-image editing on the shared ai-toolkit model.

There is no second model: ai-toolkit's MOSS transformer implements the reference
branch Moss-Image-Edit was trained with, so editing flips three flags on the
loaded holder (``is_edit``, ``kv_cache``, vision-prompt encoding) and hooks the
edit LoRA for the duration of one request.  Text-to-image and editing therefore
share one load, one copy of the weights and one VRAM budget.

This service owns the HTTP-facing concerns only: readiness checks, turning
base64 payloads into files the engine can open, and reporting status.
"""

from __future__ import annotations

import base64
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

from .edit_lora import AdapterError, adapter_signature
from .engine import EditRequest as EngineEditRequest

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class EditError(RuntimeError):
    """Editing failed: configuration, adapter or request problem."""


@dataclass
class EditRequest:
    prompt: str
    images: list[bytes]
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 4.0
    seed: int | None = None
    shift: float | None = None
    max_edge: int | None = None
    #: Extra LoRAs to stack with the edit LoRA: {name: weight}.
    loras: dict[str, float] | None = None
    progress: ProgressCallback | None = field(default=None, repr=False)


@dataclass
class EditResult:
    image: object  # PIL.Image
    width: int
    height: int
    steps: int
    seed: int
    cfg: float
    shift: float
    references: int
    seconds: float


class EditService:
    """Editing on top of the text-to-image engine (same model, same VRAM)."""

    def __init__(self, config, engine) -> None:
        if engine is None:
            raise ValueError("EditService needs the shared MossEngine instance")
        self.config = config
        self.engine = engine
        self._error: str | None = None

    # ------------------------------------------------------------ readiness
    @property
    def adapter(self) -> Path | None:
        value = getattr(self.config, "edit_adapter", "") or ""
        return Path(value) if value else None

    def availability(self) -> tuple[bool, str]:
        """(ready-to-run, reason-when-not)."""
        if not getattr(self.config, "edit_enabled", True):
            return False, "editing is disabled (edit_enabled = false)"
        adapter = self.adapter
        if adapter is None:
            return False, "edit_adapter is not configured"
        if not (adapter / "adapter_config.json").is_file():
            return False, f"edit_adapter has no adapter_config.json: {adapter}"
        if not (adapter / "adapter_model.safetensors").is_file():
            return False, f"edit_adapter has no adapter_model.safetensors: {adapter}"
        try:
            adapter_signature(adapter)
        except AdapterError as error:
            return False, str(error)
        return True, ""

    @property
    def ready(self) -> bool:
        return self.engine.ready

    def idle_seconds(self) -> float:
        return self.engine.idle_seconds()

    def status(self) -> dict:
        available, reason = self.availability()
        info = {
            "available": available,
            "reason": reason or None,
            "state": self.engine.state,
            "adapter": str(self.adapter) if self.adapter else None,
            "lora": str(self.engine.edit_lora) if self.engine.edit_lora else None,
            "backed_by": "ai-toolkit",
            "shared_model": True,
            "mode": "edit" if self.engine.edit_mode else "text-to-image",
            "load_seconds": round(self.engine.load_seconds, 2),
            "idle_seconds": round(self.idle_seconds(), 1),
            "idle_unload_seconds": getattr(self.config, "idle_unload_seconds", 0),
            "error": (self.engine.error or None) or self._error,
            "max_ref_images": int(getattr(self.config, "edit_max_ref_images", 4)),
            "max_edge": int(getattr(self.config, "edit_max_edge", 1024)),
            "device": getattr(self.config, "device", "cuda"),
            "dtype": getattr(self.config, "dtype", "bf16"),
        }
        if self.engine.ready:
            described = self.engine.describe()
            for key in ("vram_used_gib", "gpu_free", "gpu_total"):
                if key in described:
                    info[key] = described[key]
        return info

    # ------------------------------------------------------------ lifecycle
    def stop(self, reason: str = "") -> bool:
        """Drop the shared model; editing and text-to-image go idle together."""
        self._error = None
        return self.engine.unload(reason or "edit unload")

    def maybe_idle_stop(self) -> bool:
        # The shared model already has one idle timer (``idle_unload_seconds``);
        # the API watchdog calls engine.maybe_unload for it.
        return False

    # ------------------------------------------------------------- requests
    def edit(self, request: EditRequest) -> EditResult:
        available, reason = self.availability()
        if not available:
            raise EditError(reason)
        if not request.images:
            raise EditError("at least one reference image is required")

        workdir = Path(tempfile.mkdtemp(prefix=f"moss-edit-{uuid.uuid4().hex[:8]}-"))
        try:
            paths = []
            for index, payload in enumerate(request.images):
                path = workdir / f"reference-{index}.png"
                path.write_bytes(_normalise_image(payload))
                paths.append(str(path))
            engine_request = EngineEditRequest(
                prompt=request.prompt,
                references=paths,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                shift=request.shift,
                loras=request.loras,
                progress=request.progress,
            )
            try:
                image = self.engine.edit(engine_request)
            except ValueError as error:
                raise EditError(str(error)) from error
            self._error = None
            holder = getattr(self.engine.engine, "holder", None)
            shift = float(getattr(holder, "sample_shift", request.shift or self.config.edit_shift))
            return EditResult(
                image=image.image,
                width=image.image.width,
                height=image.image.height,
                steps=request.steps,
                seed=image.seed,
                cfg=request.guidance_scale,
                shift=shift,
                references=len(request.images),
                seconds=image.seconds,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _normalise_image(payload: bytes) -> bytes:
    """Re-encode as PNG so any JPEG/WebP reference goes through the same path."""
    from PIL import Image

    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception as error:  # noqa: BLE001 - reported as a request error
        raise EditError(f"reference image could not be decoded: {error}") from error


def decode_data_url(payload: str) -> bytes:
    """Accept ``data:image/png;base64,...`` and bare base64."""
    text = (payload or "").strip()
    if text.startswith("data:"):
        comma = text.find(",")
        if comma < 0:
            raise ValueError("malformed data URL")
        text = text[comma + 1 :]
    return base64.b64decode(text, validate=True)
