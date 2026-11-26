"""Reference-image editing through a resident Moss-Image-Edit worker.

The edit LoRA from https://github.com/Chenkin-x/Moss-Image-Edit needs the
*official* diffusers transformer with a reference branch, which cannot share an
interpreter with ai-toolkit's model classes.  This module owns that worker:

* it starts ``python -m moss_server.edit_worker`` on demand,
* forwards one request at a time and relays step progress,
* stops the worker after ``edit_idle_unload_seconds`` of inactivity so the VRAM
  goes back to the text-to-image engine.

The worker and its adapter are optional; when they are not installed the HTTP
layer reports the editing endpoints as unavailable instead of failing at boot.
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class EditError(RuntimeError):
    """Editing failed: configuration, worker or sampling problem."""


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
    use_kv_cache: bool = True
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
    encode_seconds: float


def _is_windows() -> bool:
    return sys.platform.startswith("win")


class EditService:
    """Owns the edit worker process and serialises requests to it."""

    def __init__(self, config, engine=None) -> None:
        self.config = config
        self.engine = engine
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._log_handle = None
        self._progress: ProgressCallback | None = None
        self._last_used = 0.0
        self._load_seconds = 0.0
        self._error: str | None = None
        self._request_id = 0
        self._started_at = 0.0
        self._status: dict = {}

    # ------------------------------------------------------------ readiness
    @property
    def repo(self) -> Path | None:
        value = getattr(self.config, "edit_repo", "") or ""
        return Path(value) if value else None

    @property
    def adapter(self) -> Path | None:
        value = getattr(self.config, "edit_adapter", "") or ""
        return Path(value) if value else None

    def availability(self) -> tuple[bool, str]:
        """(ready-to-run, reason-when-not)."""
        if not getattr(self.config, "edit_enabled", True):
            return False, "editing is disabled (edit_enabled = false)"
        repo = self.repo
        if repo is None:
            return False, "edit_repo is not configured"
        if not (repo / "moss_edit" / "model.py").is_file():
            return False, f"edit_repo does not look like a Moss-Image-Edit checkout: {repo}"
        adapter = self.adapter
        if adapter is None:
            return False, "edit_adapter is not configured"
        if not (adapter / "adapter_config.json").is_file():
            return False, f"edit_adapter has no adapter_config.json: {adapter}"
        if not any(adapter.glob("adapter_model*.safetensors")):
            return False, f"edit_adapter has no adapter_model*.safetensors: {adapter}"
        return True, ""

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def ready(self) -> bool:
        return self.alive and self._load_seconds > 0

    def idle_seconds(self) -> float:
        if not self.alive or not self._last_used:
            return 0.0
        return time.monotonic() - self._last_used

    def status(self) -> dict:
        available, reason = self.availability()
        info = {
            "available": available,
            "reason": reason or None,
            "state": "ready" if self.ready else ("loading" if self.alive else "idle"),
            "repo": str(self.repo) if self.repo else None,
            "adapter": str(self.adapter) if self.adapter else None,
            "load_seconds": round(self._load_seconds, 2),
            "idle_seconds": round(self.idle_seconds(), 1),
            "idle_unload_seconds": getattr(self.config, "edit_idle_unload_seconds", 0),
            "error": self._error,
        }
        if self._status:
            for key in ("max_ref_images", "max_edge", "device", "dtype", "vram_used_gib", "vram_total_gib"):
                if key in self._status:
                    info[key] = self._status[key]
        return info

    # ----------------------------------------------------------- lifecycle
    def _python(self) -> str:
        return getattr(self.config, "edit_python", "") or sys.executable

    def _worker_config(self) -> dict:
        config = self.config
        return {
            "repo": str(self.repo),
            "snapshot_dir": str(config.model_dir),
            "adapter": str(self.adapter),
            "device": getattr(config, "device", "cuda"),
            "dtype": "bfloat16" if getattr(config, "dtype", "bf16") in ("bf16", "float16") else "bfloat16",
            "vae_dtype": "float32",
            "max_sequence_length": int(getattr(config, "default_max_sequence_length", 4096)),
            "max_edge": int(getattr(config, "edit_max_edge", 1024)),
            "max_ref_images": int(getattr(config, "edit_max_ref_images", 4)),
            "shift": float(getattr(config, "edit_shift", 5.0)),
            "training_shift": 1.0,
        }

    def start(self) -> None:
        """Spawn the worker and wait until it reports ready."""
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self.alive:
            return
        available, reason = self.availability()
        if not available:
            raise EditError(reason)

        if self.engine is not None and getattr(self.config, "edit_free_t2i", True):
            try:
                if self.engine.ready:
                    LOGGER.info("Unloading the text-to-image engine to make room for editing")
                    self.engine.unload("edit worker start")
            except Exception:  # noqa: BLE001 - never block editing on this
                LOGGER.exception("could not unload the text-to-image engine")

        config_path = Path(self.config.output_dir) / ".edit-worker.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._worker_config()
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self._queue = queue.Queue()
        self._error = None
        self._load_seconds = 0.0
        self._status = {}
        self._started_at = time.perf_counter()

        log_path = Path(self.config.output_dir) / ".edit-worker.log"
        self._log_handle = open(log_path, "a", encoding="utf-8", errors="replace")
        self._log_handle.write(f"\n=== worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self._log_handle.flush()

        root = Path(__file__).resolve().parents[1]
        env = dict(**__import__("os").environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(root) + (
            __import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        creationflags = 0
        if _is_windows():
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        LOGGER.info("Starting edit worker: %s (repo=%s)", self._python(), self.repo)
        try:
            self._proc = subprocess.Popen(
                [self._python(), "-m", "moss_server.edit_worker", "--config", str(config_path)],
                cwd=str(root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creationflags,
            )
        except Exception as error:  # noqa: BLE001 - surfaced as EditError
            raise EditError(f"could not start the edit worker: {error}") from error

        self._reader = threading.Thread(target=self._read_loop, name="edit-worker-reader", daemon=True)
        self._reader.start()

        timeout = float(getattr(self.config, "edit_timeout", 900.0))
        try:
            message = self._queue.get(timeout=timeout)
        except queue.Empty as error:
            self._kill_locked("startup timed out")
            raise EditError(f"edit worker did not become ready within {timeout:.0f}s") from error
        if message.get("event") == "failed" or not message.get("ok"):
            detail = message.get("error") or "unknown error"
            self._kill_locked("startup failed")
            raise EditError(f"edit worker failed to load: {detail}")
        self._status = message
        self._load_seconds = float(message.get("load_seconds") or time.perf_counter() - self._started_at)
        self._last_used = time.monotonic()
        LOGGER.info("Edit worker ready in %.1fs", self._load_seconds)

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("edit worker wrote a non-JSON line: %s", line[:200])
                continue
            if message.get("event") == "progress":
                callback = self._progress
                if callback is not None:
                    try:
                        callback(int(message.get("step", 0)), int(message.get("total", 0)))
                    except Exception:  # noqa: BLE001 - progress must not break the worker
                        LOGGER.exception("edit progress callback failed")
                continue
            self._queue.put(message)
        # stdout closed: the process is gone.
        self._queue.put({"ok": False, "event": "exit", "error": "edit worker exited"})

    def stop(self, reason: str = "") -> bool:
        with self._lock:
            return self._kill_locked(reason)

    def _kill_locked(self, reason: str = "") -> bool:
        proc = self._proc
        if proc is None:
            return False
        LOGGER.info("Stopping edit worker%s", f" ({reason})" if reason else "")
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps({"id": 0, "cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                except Exception:  # noqa: BLE001 - best effort
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except Exception:  # noqa: BLE001
                    pass
                self._log_handle = None
            self._proc = None
            self._reader = None
            self._load_seconds = 0.0
            self._status = {}
            self._last_used = 0.0
        return True

    def maybe_idle_stop(self) -> bool:
        idle = float(getattr(self.config, "edit_idle_unload_seconds", 0) or 0)
        if idle <= 0 or not self.alive:
            return False
        if self.idle_seconds() < idle:
            return False
        return self.stop(f"idle for {idle:.0f}s")

    # ------------------------------------------------------------- requests
    def edit(self, request: EditRequest) -> EditResult:
        with self._lock:
            self._start_locked()
            if self._proc is None or self._proc.stdin is None:
                raise EditError("edit worker is not running")
            self._request_id += 1
            request_id = self._request_id
            self._progress = request.progress
            self._last_used = time.monotonic()
            payload = {
                "id": request_id,
                "cmd": "edit",
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "images": [base64.b64encode(image).decode("ascii") for image in request.images],
                "width": int(request.width),
                "height": int(request.height),
                "steps": int(request.steps),
                "cfg": float(request.guidance_scale),
                "seed": request.seed,
                "shift": request.shift,
                "max_edge": request.max_edge,
                "use_kv_cache": bool(request.use_kv_cache),
            }
            try:
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except Exception as error:  # noqa: BLE001
                self._kill_locked("write failed")
                raise EditError(f"could not send the request to the edit worker: {error}") from error

            timeout = float(getattr(self.config, "edit_timeout", 900.0))
            try:
                message = self._queue.get(timeout=timeout)
            except queue.Empty as error:
                self._kill_locked("request timed out")
                raise EditError(f"edit request timed out after {timeout:.0f}s") from error
            finally:
                self._progress = None
                self._last_used = time.monotonic()

            if message.get("event") == "exit":
                self._kill_locked("worker exited")
                raise EditError(message.get("error") or "edit worker exited")
            if not message.get("ok"):
                raise EditError(message.get("error") or "edit failed")

        from PIL import Image

        image = Image.open(BytesIO(base64.b64decode(message["image"]))).convert("RGB")
        return EditResult(
            image=image,
            width=int(message.get("width", request.width)),
            height=int(message.get("height", request.height)),
            steps=int(message.get("steps", request.steps)),
            seed=int(message.get("seed", request.seed or 0)),
            cfg=float(message.get("cfg", request.guidance_scale)),
            shift=float(message.get("shift", request.shift or 0.0)),
            references=int(message.get("references", len(request.images))),
            seconds=float(message.get("seconds", 0.0)),
            encode_seconds=float(message.get("encode_seconds", 0.0)),
        )
