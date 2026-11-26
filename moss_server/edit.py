"""Reference-image editing on the ai-toolkit engine.

Editing shares the text-to-image stack: the same ai-toolkit engine loads the
model, the same convrot8 quantisation and layer offloading apply, and the same
lazy-load / idle-unload lifecycle keeps VRAM in check.  The only additions are
the edit LoRA (fused once by :mod:`moss_server.edit_lora`) and reference images
passed to the model as clean latents.

The service therefore owns a second :class:`~moss_server.engine.MossEditEngine`
and serialises it against the text-to-image engine: both want the card, so a
request on one side unloads the other when ``edit_free_t2i`` is enabled.
"""

from __future__ import annotations

import base64
import logging
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

from .edit_lora import AdapterError, adapter_signature
from .engine import EditRequest as EngineEditRequest
from .engine import MossEditEngine

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class EditError(RuntimeError):
    """Editing failed: configuration, adapter or sampling problem."""


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
    """Owns the edit engine and turns HTTP requests into engine calls."""

    def __init__(self, config, engine=None) -> None:
        self.config = config
        self.t2i = engine
        self._lock = threading.Lock()
        self._engine: MossEditEngine | None = None
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
    def engine(self) -> MossEditEngine | None:
        return self._engine

    @property
    def ready(self) -> bool:
        return self._engine is not None and self._engine.ready

    def idle_seconds(self) -> float:
        return self._engine.idle_seconds() if self._engine is not None else 0.0

    def status(self) -> dict:
        available, reason = self.availability()
        engine = self._engine
        state = engine.state if engine is not None else "idle"
        info = {
            "available": available,
            "reason": reason or None,
            "state": state,
            "adapter": str(self.adapter) if self.adapter else None,
            "lora": str(engine.fused_lora) if engine and engine.fused_lora else None,
            "backed_by": "ai-toolkit",
            "load_seconds": round(engine.load_seconds, 2) if engine else 0.0,
            "idle_seconds": round(self.idle_seconds(), 1),
            "idle_unload_seconds": getattr(self.config, "edit_idle_unload_seconds", 0),
            "error": (engine.error if engine else None) or self._error,
            "max_ref_images": int(getattr(self.config, "edit_max_ref_images", 4)),
            "max_edge": int(getattr(self.config, "edit_max_edge", 1024)),
            "device": getattr(self.config, "device", "cuda"),
            "dtype": getattr(self.config, "dtype", "bf16"),
        }
        if engine is not None and engine.ready:
            described = engine.describe()
            for key in ("vram_used_gib", "gpu_free", "gpu_total"):
                if key in described:
                    info[key] = described[key]
        return info

    # ------------------------------------------------------------ lifecycle
    def _ensure_engine(self) -> MossEditEngine:
        with self._lock:
            if self._engine is None:
                self._engine = MossEditEngine(self.config)
            return self._engine

    def _free_t2i(self) -> None:
        if self.t2i is None or not getattr(self.config, "edit_free_t2i", True):
            return
        try:
            if self.t2i.ready:
                LOGGER.info("Unloading the text-to-image engine to make room for editing")
                self.t2i.unload("edit request")
        except Exception:  # noqa: BLE001 - never block editing on this
            LOGGER.exception("could not unload the text-to-image engine")

    def stop(self, reason: str = "") -> bool:
        engine = self._engine
        if engine is None:
            return False
        try:
            return engine.unload(reason or "edit unload")
        finally:
            self._error = None

    def maybe_idle_stop(self) -> bool:
        idle = float(getattr(self.config, "edit_idle_unload_seconds", 0) or 0)
        engine = self._engine
        if engine is None or idle <= 0 or engine.state != "ready":
            return False
        if engine.idle_seconds() < idle:
            return False
        return engine.maybe_unload(idle)

    # ------------------------------------------------------------- requests
    def edit(self, request: EditRequest) -> EditResult:
        available, reason = self.availability()
        if not available:
            raise EditError(reason)
        if not request.images:
            raise EditError("at least one reference image is required")

        self._free_t2i()
        engine = self._ensure_engine()
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
                progress=request.progress,
            )
            try:
                image = engine.edit(engine_request)
            except ValueError as error:
                raise EditError(str(error)) from error
            self._error = None
            holder = getattr(engine.engine, "holder", None)
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
