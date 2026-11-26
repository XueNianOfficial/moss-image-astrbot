"""Resident worker process for reference-image editing (Moss-Image-Edit).

Text-to-image runs on ai-toolkit's engine inside the server process; editing
needs the *official* diffusers MOSS-Image transformer plus the reference-image
branch from https://github.com/Chenkin-x/Moss-Image-Edit.  That cannot share an
interpreter with ai-toolkit's own model classes, so it lives in this small
worker: the parent starts it on demand, sends one JSON request per line and
kills it after an idle period to give the VRAM back.

Wire protocol (JSON per line):

    parent -> worker   {"id": "...", "cmd": "edit", "images": ["<b64 png>"], ...}
    worker -> parent   {"id": "...", "event": "progress", "step": 3, "total": 20}
    worker -> parent   {"id": "...", "ok": true, "image": "<b64 png>", ...}
    parent -> worker   {"id": "...", "cmd": "ping"}
    parent -> worker   {"id": "...", "cmd": "shutdown"}

stdout carries protocol lines only; human-readable logs go to stderr.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
import traceback
from pathlib import Path

#: The editing pipeline was trained with the official MOSS-Image snapshot, so
#: this worker always loads <snapshot_dir>/pipeline_moss_image.py.
PROTOCOL_VERSION = 1


def _log(message: str) -> None:
    print(f"[edit-worker] {message}", file=sys.stderr, flush=True)


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _dtype(name: str):
    import torch

    table = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in table:
        raise ValueError(f"unsupported dtype {name!r}")
    return table[name]


def _aligned_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """Match moss_edit.processing.resize_image: cap the long edge, snap to /16."""
    limit = max(16, max_edge // 16 * 16)
    scale = min(1.0, limit / max(width, height))

    def align(value: float) -> int:
        return min(limit, max(16, int(math.floor(value * scale / 16 + 0.5)) * 16))

    return align(width), align(height)


def read_adapter_options(adapter: str | Path) -> tuple[dict, dict]:
    """rank/alpha/dropout/dtype/targets from an adapter_config.json bundle."""
    config = json.loads((Path(adapter).expanduser() / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("format") != "moss-edit-lora" or config.get("format_version") != 1:
        raise ValueError("unsupported LoRA adapter format/version")
    modules, targets = config.get("modules"), config.get("targets")
    if not isinstance(modules, dict) or not modules or targets != sorted(modules):
        raise ValueError("adapter must contain consistent, nonempty module/target metadata")
    keys = ("rank", "alpha", "dropout", "param_dtype")
    first = next(iter(modules.values()))
    if not isinstance(first, dict) or not all(key in first for key in keys):
        raise ValueError("adapter is missing rank/alpha/dropout/param_dtype")
    shared = {key: first[key] for key in keys}
    for value in modules.values():
        if not isinstance(value, dict) or any(value.get(key) != shared[key] for key in keys):
            raise ValueError("this sampler requires uniform rank/alpha/dropout/dtype across targets")
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("adapter metadata must be a JSON object")
    return {**shared, "targets": sorted(modules)}, metadata


class EditWorker:
    """Owns the encoder, the LoRA-injected transformer and the KV-cache wrapper."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.snapshot = Path(config["snapshot_dir"]).expanduser().resolve(strict=True)
        self.adapter = Path(config["adapter"]).expanduser().resolve(strict=True)
        self.device = config.get("device", "cuda")
        self.dtype = _dtype(config.get("dtype", "bfloat16"))
        self.vae_dtype = _dtype(config.get("vae_dtype", "float32"))
        self.max_sequence_length = int(config.get("max_sequence_length", 4096))
        self.max_edge = int(config.get("max_edge", 1024))
        self.max_ref_images = int(config.get("max_ref_images", 4))
        self.default_shift = float(config.get("shift", 5.0))
        self.training_shift = float(config.get("training_shift", 1.0))
        self._condition_cache: dict[tuple, object] = {}
        self.load_seconds = 0.0

        # Loaded in load(); kept as attributes so requests can reuse them.
        self.torch = None
        self.data = None
        self.sampling = None
        self.base = None
        self.editor = None
        self.scheduler = None
        self.vae = None
        self.prompt_components = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        started = time.perf_counter()
        repo = str(self.config["repo"])
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import torch

        import moss_edit.processing as data
        from moss_edit import components as adapter
        from moss_edit import lora as lora_module
        from moss_edit.model import MossEditModel
        from moss_edit.sampling import sample_latents

        self.torch = torch
        self.data = data
        self.sampling = sample_latents

        _log("loading text encoder + VAE (frozen)")
        encoding = data.load_encoding_components(
            self.snapshot,
            device=self.device,
            dtype=self.dtype,
            vae_dtype=self.vae_dtype,
        )
        self.prompt_components = encoding.prompt_components
        # Encoders stay in RAM and only visit the GPU while they encode.
        self.prompt_components.offload_text_encoder = True
        self.vae = encoding.vae.to("cpu")

        options, metadata = read_adapter_options(self.adapter)
        options["param_dtype"] = _dtype(options["param_dtype"])
        _log(
            f"loading transformer + edit LoRA (rank {options['rank']}, "
            f"{len(options['targets'])} modules, dit_quantization={metadata.get('dit_quantization', 'none')})"
        )
        self.base = adapter.load_transformer(self.snapshot, device=self.device, dtype=self.dtype)
        lora_module.inject_lora(self.base, **options)
        lora_module.load_adapter(self.base, self.adapter)
        self.base.requires_grad_(False).eval()
        self.editor = MossEditModel(self.base, gradient_checkpointing=False).eval()

        module = adapter.load_moss_module(self.snapshot)
        self.scheduler = module.MOSSImageScheduler.from_pretrained(
            str(self.snapshot / "scheduler"),
            local_files_only=True,
            shift=self.default_shift,
            training_shift=self.training_shift,
        )
        self.load_seconds = time.perf_counter() - started
        _log(f"ready in {self.load_seconds:.1f}s")

    # -------------------------------------------------------------- encoding
    def _conditions(self, prompt: str, negative: str, cfg: float):
        import moss_edit.components as components

        key = (prompt, negative if cfg != 1 else None)
        cached = self._condition_cache.get(key)
        if cached is not None:
            return cached
        conditions = components.encode_prompts(
            self.prompt_components,
            [prompt],
            negative_prompts=[negative] if cfg != 1 else None,
            max_sequence_length=self.max_sequence_length,
            device="cpu",
        )
        if len(self._condition_cache) >= 8:
            self._condition_cache.clear()
        self._condition_cache[key] = conditions
        return conditions

    def _reference_latents(self, images: list[str], max_edge: int):
        from PIL import Image

        torch = self.torch
        latents = []
        self.vae.to(self.device)
        try:
            for payload in images:
                raw = base64.b64decode(payload, validate=False)
                with Image.open(io.BytesIO(raw)) as opened:
                    opened.load()
                    image = opened.convert("RGB")
                width, height = _aligned_size(image.width, image.height, max_edge)
                if (width, height) != image.size:
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                tensor = self.data.image_to_tensor(image)
                with torch.no_grad():
                    latent = self.data.encode_image(self.vae, tensor, sample_posterior=False)
                latents.append(latent.detach().cpu())
        finally:
            self.vae.to("cpu")
        return latents

    # ------------------------------------------------------------------ edit
    def edit(self, request: dict, progress=None) -> dict:
        started = time.perf_counter()
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        images = request.get("images") or []
        if not isinstance(images, list) or not 1 <= len(images) <= self.max_ref_images:
            raise ValueError(f"provide 1..{self.max_ref_images} reference images")
        steps = int(request.get("steps") or 20)
        if not 1 <= steps <= 200:
            raise ValueError("steps must be within 1..200")
        cfg = float(request.get("cfg", 4.0))
        if not math.isfinite(cfg) or cfg < 0:
            raise ValueError("cfg must be a finite nonnegative number")
        height, width = int(request["height"]), int(request["width"])
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16")
        seed = int(request.get("seed") or 0)
        shift = request.get("shift")
        max_edge = int(request.get("max_edge") or self.max_edge)
        use_kv_cache = bool(request.get("use_kv_cache", True))

        encode_started = time.perf_counter()
        conditions = self._conditions(prompt, str(request.get("negative_prompt") or ""), cfg)
        references = self._reference_latents(images, max_edge)
        encode_seconds = time.perf_counter() - encode_started

        if shift is not None:
            self.scheduler.config.shift = float(shift)
        elif self.scheduler.config.shift != self.default_shift:
            self.scheduler.config.shift = self.default_shift

        sample_started = time.perf_counter()

        def _progress(done: int, total: int) -> None:
            _log(f"step {done}/{total} (+{time.perf_counter() - sample_started:.2f}s)")
            if progress is not None:
                progress(done, total)

        latents = self.sampling(
            self.editor,
            self.scheduler,
            references,
            conditions.positive,
            conditions.negative,
            height=height,
            width=width,
            steps=steps,
            cfg=cfg,
            seed=seed,
            device=self.device,
            latent_channels=int(self.base.config.latent_channels),
            use_kv_cache=use_kv_cache,
            progress=_progress,
        )
        sample_seconds = time.perf_counter() - sample_started
        _log(f"sampled {steps} steps in {sample_seconds:.1f}s")
        self.vae.to(self.device)
        try:
            with self.torch.no_grad():
                decoded = self.data.decode_latents(self.vae, latents)
        finally:
            self.vae.to("cpu")
        image = self._to_png(decoded)
        return {
            "image": image,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "shift": float(self.scheduler.config.shift),
            "references": len(references),
            "encode_seconds": round(encode_seconds, 2),
            "sample_seconds": round(sample_seconds, 2),
            "seconds": round(time.perf_counter() - started, 2),
            "kv_cache": use_kv_cache,
        }

    def _to_png(self, decoded) -> str:
        from PIL import Image

        tensor = decoded[0].detach().float().cpu().clamp(-1, 1)
        pixels = ((tensor + 1) * 127.5).round().to(self.torch.uint8).permute(1, 2, 0).numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        info = {
            "loaded": True,
            "snapshot_dir": str(self.snapshot),
            "adapter": str(self.adapter),
            "device": self.device,
            "dtype": self.config.get("dtype", "bfloat16"),
            "max_ref_images": self.max_ref_images,
            "max_edge": self.max_edge,
            "load_seconds": round(self.load_seconds, 2),
            "protocol": PROTOCOL_VERSION,
        }
        torch = self.torch
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["vram_used_gib"] = round((total - free) / 2**30, 2)
            info["vram_total_gib"] = round(total / 2**30, 2)
        return info


def serve(worker: EditWorker) -> int:
    """Read one JSON request per line until stdin closes."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = request.get("cmd", "edit")
            if command == "shutdown":
                _emit({"id": request_id, "ok": True, "bye": True})
                return 0
            if command == "ping":
                _emit({"id": request_id, "ok": True, **worker.status()})
                continue
            if command != "edit":
                raise ValueError(f"unknown command {command!r}")

            def progress(done: int, total: int, _id=request_id) -> None:
                _emit({"id": _id, "event": "progress", "step": done, "total": total})

            result = worker.edit(request, progress=progress)
            _emit({"id": request_id, "ok": True, **result})
        except Exception as error:  # noqa: BLE001 - reported to the parent
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Moss-Image-Edit resident worker")
    parser.add_argument("--config", required=True, help="path to the worker JSON config")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        worker = EditWorker(config)
        worker.load()
    except Exception as error:  # noqa: BLE001 - parent surfaces the message
        traceback.print_exc(file=sys.stderr)
        _emit({"event": "failed", "ok": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    _emit({"event": "ready", "ok": True, **worker.status()})
    return serve(worker)


if __name__ == "__main__":
    raise SystemExit(main())
