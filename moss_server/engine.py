"""Model engine backed by ai-toolkit's inference engine.

Instead of re-implementing loading, quantisation and sampling, this delegates to
ai-toolkit's own MOSS-Image holder:

* ``extensions_built_in/inference_engine/engine.py`` — the resident engine: it
  owns the model holder, the component pool, the per-request LoRA stack and the
  generation queue.
* ``extensions_built_in/diffusion_models/moss_image`` — the MOSS-Image model
  (convrot8 int8 quantisation, layer offloading, Qwen3.5 text encoder).
* ``toolkit/inference_lora.py`` — hook-mode LoRA, so LoRA weights are attached
  dynamically and never merged into the base weights.

The server keeps its own lifecycle on top: lazy load on the first request,
idle unload after ``idle_unload_seconds``, and one generation at a time.
"""
from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from .config import Config
from .lora import LoRARegistry

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    height: int = 1024
    width: int = 1024
    steps: int = 30
    guidance_scale: float = 3.0
    seed: int | None = None
    num_images: int = 1
    loras: dict[str, float] | None = None
    #: Flow-matching sigma shift; None keeps the holder's configured value.
    shift: float | None = None
    #: CFG skip blocks; None keeps the holder's configured value.
    uncond_skip_layers: tuple[int, ...] | None = None
    progress: ProgressCallback | None = None


@dataclass
class GeneratedImage:
    image: Image.Image
    seed: int
    seconds: float


@dataclass
class EngineStats:
    requests: int = 0
    images: int = 0
    total_seconds: float = 0.0
    last_seconds: float = 0.0
    last_steps: int = 0
    last_size: tuple[int, int] = (0, 0)
    loads: int = 0
    unloads: int = 0


def gpu_memory() -> dict:
    """Free/total VRAM in GiB, preferring nvidia-smi over torch on WDDM."""
    info = {"free": None, "total": None, "source": None}
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        free_mib, total_mib = (int(part) for part in output.split(","))
        info.update(free=free_mib / 1024, total=total_mib / 1024, source="nvidia-smi")
        return info
    except Exception:  # noqa: BLE001 - fall back to torch
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update(free=free / 1024**3, total=total / 1024**3, source="torch")
    return info


class MossEngine:
    """Drives ai-toolkit's engine for one MOSS-Image checkpoint."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = "idle"
        self.error: str | None = None
        self.state_lock = threading.Lock()   # guards load/unload transitions
        self.lock = threading.RLock()        # one generation at a time
        self.last_used = time.monotonic()
        self.load_seconds = 0.0
        self.stats = EngineStats()
        self._warmed = False

        self.engine = None
        self._thread: threading.Thread | None = None
        self.last_status = ""

        self.loras = LoRARegistry()
        for path in config.loras:
            try:
                self.loras.load(Path(path).stem, path, scale=config.lora_scale)
            except (OSError, ValueError) as error:
                LOGGER.warning("startup LoRA %s not loaded: %s", path, error)

    # --------------------------------------------------------------- status
    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_used

    def _status(self, message: str) -> None:
        self.last_status = str(message)
        LOGGER.debug("[aitk] %s", message)

    # ---------------------------------------------------------------- ai-toolkit
    def _engine_class(self):
        root = str(Path(self.config.aitk_root))
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from extensions_built_in.inference_engine.engine import Engine  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - broken checkout
            raise RuntimeError(
                f"cannot import ai-toolkit's inference engine from {root}: {error}"
            ) from error
        return Engine

    def _model_spec(self, loras: list[dict] | None = None) -> dict:
        cfg = self.config
        spec = {
            "arch": "moss_image",
            "name_or_path": cfg.model_dir,
            "dtype": cfg.dtype,
            "quantize": cfg.quantize,
            "qtype": cfg.qtype,
            "quantize_te": cfg.quantize_te,
            "qtype_te": cfg.qtype_te,
            "low_vram": cfg.low_vram,
            "layer_offloading": cfg.layer_offloading,
            "layer_offloading_transformer_percent": cfg.layer_offloading_transformer_percent,
            "layer_offloading_text_encoder_percent": cfg.layer_offloading_text_encoder_percent,
            "lora_mode": cfg.lora_mode,
            "loras": loras if loras is not None else self.loras.specs(),
            "model_kwargs": {
                "uncond_skip_layers": list(cfg.default_uncond_skip_layers),
                "sample_shift": float(cfg.default_shift),
                "max_text_length": int(cfg.default_max_sequence_length),
            },
        }
        if cfg.vae_path:
            spec["vae_path"] = cfg.vae_path
        return spec

    # ---------------------------------------------------------------- load
    def ensure_loaded(self) -> bool:
        """Load the model if needed. Returns True when this call loaded it."""
        if self.state == "ready":
            self.touch()
            return False
        with self.state_lock:
            if self.state == "ready":
                self.touch()
                return False
            self.state = "loading"
            self.error = None
            LOGGER.info("Loading MOSS-Image through ai-toolkit from %s", self.config.model_dir)
            try:
                self._load()
            except Exception as error:  # noqa: BLE001 - reported to the caller
                self.state = "error"
                self.error = f"{type(error).__name__}: {error}"
                self._release()
                LOGGER.exception("Model loading failed")
                raise
            self.state = "ready"
            self.touch()
            self.stats.loads += 1
        if self.config.warmup and not self._warmed:
            self._warmed = True
            self.warmup()
        return True

    def _load(self) -> None:
        started = time.perf_counter()
        Engine = self._engine_class()
        output_folder = Path(self.config.output_dir) / "engine"
        output_folder.mkdir(parents=True, exist_ok=True)

        self.engine = Engine(
            device=self.config.device,
            output_folder=str(output_folder),
            status_cb=self._status,
        )
        self._thread = threading.Thread(target=self.engine.run_forever, name="aitk-engine", daemon=True)
        self._thread.start()

        job = self.engine.preload(self._model_spec())
        if job.status != "done":
            raise RuntimeError(job.error or "ai-toolkit could not load the model")

        self.load_seconds = time.perf_counter() - started
        LOGGER.info(
            "Model ready in %.1fs | %s | quantize=%s/%s low_vram=%s offload(te)=%.0f%%",
            self.load_seconds, self.config.model_dir,
            self.config.quantize, self.config.qtype,
            self.config.low_vram, self.config.layer_offloading_text_encoder_percent * 100,
        )

    def _release(self) -> None:
        engine, thread = self.engine, self._thread
        self.engine = None
        self._thread = None
        if engine is not None:
            try:
                engine.stop("unload")
            except Exception:  # noqa: BLE001 - best effort
                LOGGER.debug("engine.stop failed", exc_info=True)
            if thread is not None and thread.is_alive():
                thread.join(timeout=30)
            try:
                engine.unload()
            except Exception:  # noqa: BLE001 - pool may already be gone
                LOGGER.debug("engine.unload failed", exc_info=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self, reason: str = "manual") -> bool:
        with self.state_lock:
            if self.state != "ready":
                return False
            self._release()
            self.state = "idle"
            self.stats.unloads += 1
            LOGGER.info("Model unloaded (%s); VRAM released", reason)
            return True

    def maybe_unload(self, idle_seconds: float) -> bool:
        """Unload when idle and no generation is running (never blocks)."""
        if self.state != "ready" or self.idle_seconds() < idle_seconds:
            return False
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if not self.state_lock.acquire(blocking=False):
                return False
            try:
                if self.state != "ready" or self.idle_seconds() < idle_seconds:
                    return False
                self._release()
                self.state = "idle"
                self.stats.unloads += 1
                LOGGER.info("Model unloaded after %.0fs idle; VRAM released", idle_seconds)
                return True
            finally:
                self.state_lock.release()
        finally:
            self.lock.release()

    # -------------------------------------------------------------- generate
    def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        with self.lock:
            # An unknown name would otherwise be ignored and silently drop the
            # caller's LoRA, so fail loudly instead.
            if request.loras:
                missing = [name for name in request.loras if name not in self.loras]
                if missing:
                    raise ValueError(
                        f"LoRA not registered: {', '.join(missing)} (POST /v1/loras/load first)"
                    )
            self.ensure_loaded()
            self._apply_sampling_overrides(request)
            started = time.perf_counter()
            model = self._model_spec(self.loras.specs(request.loras))
            seeds = self._seeds(request)
            images: list[GeneratedImage] = []
            for seed in seeds:
                sample = {
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt or "",
                    "width": int(request.width),
                    "height": int(request.height),
                    "num_inference_steps": int(request.steps),
                    "guidance_scale": float(request.guidance_scale),
                    "seed": int(seed),
                    "output_ext": "png",
                }
                job = self.engine.submit(model, sample, {"latents": "none"})
                self._wait(job, request.progress)
                images.append(self._collect(job, seed))

            total = time.perf_counter() - started
            self.stats.requests += 1
            self.stats.images += len(images)
            self.stats.total_seconds += total
            self.stats.last_seconds = total
            self.stats.last_steps = request.steps
            self.stats.last_size = (request.width, request.height)
            self.touch()
            LOGGER.info(
                "Generated %d image(s) %dx%d, %d steps in %.1fs",
                len(images), request.width, request.height, request.steps, total,
            )
            return images

    def _apply_sampling_overrides(self, request: GenerationRequest) -> None:
        """Push per-request sampler settings onto the live ai-toolkit holder.

        ``sample_shift`` and ``uncond_skip_layers`` are plain attributes on the
        holder that ``generate_single_image`` reads on every call, so overriding
        them here is cheap and needs no reload (and no model-key change).
        """
        holder = getattr(self.engine, "holder", None)
        if holder is None:
            return
        if request.shift is not None and hasattr(holder, "sample_shift"):
            holder.sample_shift = float(request.shift)
        if request.uncond_skip_layers is not None and hasattr(holder, "uncond_skip_layers"):
            holder.uncond_skip_layers = tuple(int(i) for i in request.uncond_skip_layers)

    def _wait(self, job, progress: ProgressCallback | None) -> None:
        deadline = time.monotonic() + self.config.generation_timeout
        last_step = -1
        while job.status in ("queued", "running"):
            step = int(getattr(job, "step", 0) or 0)
            total = int(getattr(job, "total_steps", 0) or 0)
            if progress and total and step != last_step:
                last_step = step
                progress(step, total)
            if time.monotonic() > deadline:
                if self.engine is not None:
                    self.engine.cancel(job.request_id)
                raise TimeoutError(
                    f"generation exceeded {self.config.generation_timeout:.0f}s"
                )
            time.sleep(0.05)
        if job.status != "done":
            raise RuntimeError(job.error or f"generation ended as {job.status}")

    def _collect(self, job, seed: int) -> GeneratedImage:
        results = getattr(job, "results", None) or []
        if not results:
            raise RuntimeError("generation produced no image")
        path = Path(results[0]["path"])
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        shutil.rmtree(path.parent, ignore_errors=True)
        return GeneratedImage(image=image, seed=seed, seconds=float(results[0].get("seconds") or 0.0))

    def _seeds(self, request: GenerationRequest) -> list[int]:
        if request.seed is not None:
            return [int(request.seed) + offset for offset in range(request.num_images)]
        generator = torch.Generator(device="cpu")
        generator.seed()
        return [int(torch.randint(0, 2**31 - 1, (1,), generator=generator)) for _ in range(request.num_images)]

    # ------------------------------------------------------------------ misc
    def warmup(self) -> float:
        """One tiny generation so the first real request is not the slow one."""
        if not self.ready:
            return 0.0
        started = time.perf_counter()
        try:
            self.generate(
                GenerationRequest(
                    prompt="warmup", height=256, width=256, steps=2,
                    guidance_scale=1.0, seed=0, num_images=1,
                )
            )
        except Exception as error:  # noqa: BLE001 - warmup must never block startup
            LOGGER.warning("Warmup failed (server still usable): %s", error)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return time.perf_counter() - started

    def describe(self) -> dict:
        cfg = self.config
        info = {
            "engine": "ai-toolkit",
            "state": self.state,
            "error": self.error,
            "model_dir": cfg.model_dir,
            "arch": "moss_image",
            "quantize": cfg.quantize,
            "qtype": cfg.qtype,
            "quantize_te": cfg.quantize_te,
            "qtype_te": cfg.qtype_te,
            "low_vram": cfg.low_vram,
            "layer_offloading": cfg.layer_offloading,
            "lora_mode": cfg.lora_mode,
            "loras": self.loras.list(),
            "load_seconds": round(self.load_seconds, 2),
            "idle_seconds": round(self.idle_seconds(), 1),
            "idle_unload_seconds": cfg.idle_unload_seconds,
            "model_meta": {
                "arch": "moss_image",
                "base": cfg.model_dir,
                "quantisation": cfg.qtype if cfg.quantize else "bf16",
                "loras": [{"name": e["name"], "scale": e["scale"]} for e in self.loras.list()],
            },
            "last_status": self.last_status,
        }
        if self.engine is not None:
            try:
                health = self.engine.health()
            except Exception:  # noqa: BLE001 - health must not break status
                health = None
            if health:
                vram = health.get("vram") or {}
                if vram.get("used"):
                    info["vram_used_gib"] = round(vram["used"] / 1024**3, 2)
                info["engine_busy"] = health.get("busy")
                info["engine_stats"] = health.get("stats")
                info["active"] = health.get("active")
        info.update({f"gpu_{key}": value for key, value in gpu_memory().items()})
        info["stats"] = {
            "requests": self.stats.requests,
            "images": self.stats.images,
            "loads": self.stats.loads,
            "unloads": self.stats.unloads,
            "total_seconds": round(self.stats.total_seconds, 1),
            "last_seconds": round(self.stats.last_seconds, 1),
            "last_steps": self.stats.last_steps,
            "last_size": self.stats.last_size,
        }
        return info
