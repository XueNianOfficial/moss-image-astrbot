"""FastAPI application: OpenAI-compatible images API plus MOSS-Image controls.

Lifecycle
---------
The model is *not* loaded at startup by default.  The first generation request
loads it (``engine.ensure_loaded``), and a watchdog releases it again after
``idle_unload_seconds`` of inactivity so an idle server holds no VRAM.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Config
from .engine import MossEngine
from .lora import scan_lora_dirs
from .schemas import (
    ImageData,
    ImageGenerationRequest,
    ImagesResponse,
    LoRALoadRequest,
    LoRAScaleRequest,
    NativeGenerationRequest,
    build_generation_request,
)

LOGGER = logging.getLogger(__name__)


def _save_image(image, config: Config) -> tuple[str, str]:
    day = datetime.now().strftime("%Y-%m-%d")
    directory = Path(config.output_dir) / day
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%H%M%S}-{uuid.uuid4().hex[:8]}.png"
    path = directory / filename
    image.save(path, "PNG")
    relative = f"{day}/{filename}"
    return relative, f"{config.public_base_url}/files/{relative}"


def _encode_response(images, response_format: str, config: Config, seconds: float) -> ImagesResponse:
    payload: list[ImageData] = []
    for generated in images:
        if response_format == "b64_json":
            buffer = BytesIO()
            generated.image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            payload.append(
                ImageData(
                    b64_json=encoded,
                    seed=generated.seed,
                    width=generated.image.width,
                    height=generated.image.height,
                    seconds=round(generated.seconds, 2),
                )
            )
        else:
            _, url = _save_image(generated.image, config)
            payload.append(
                ImageData(
                    url=url,
                    seed=generated.seed,
                    width=generated.image.width,
                    height=generated.image.height,
                    seconds=round(generated.seconds, 2),
                )
            )
    return ImagesResponse(
        created=int(time.time()),
        data=payload,
        seed=images[0].seed if images else None,
        seconds=round(seconds, 2),
    )


def create_app(config: Config) -> FastAPI:
    engine = MossEngine(config)

    async def _watchdog(stop: asyncio.Event) -> None:
        """Release the model after the configured idle period."""
        idle = config.idle_unload_seconds
        if idle <= 0:
            return
        interval = max(5.0, min(30.0, idle / 4))
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            if engine.state != "ready" or engine.idle_seconds() < idle:
                continue
            try:
                if await asyncio.to_thread(engine.maybe_unload, idle):
                    LOGGER.info("Idle for %.0fs: model unloaded", idle)
            except Exception:  # noqa: BLE001 - watchdog must survive
                LOGGER.exception("idle unload failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.semaphore = asyncio.Semaphore(1)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        stop = asyncio.Event()
        watchdog = asyncio.create_task(_watchdog(stop))

        eager = None
        if not config.lazy_load:
            def _preload() -> None:
                try:
                    engine.ensure_loaded()
                except Exception:  # noqa: BLE001 - reported through /v1/status
                    LOGGER.exception("preload failed")

            eager = asyncio.create_task(asyncio.to_thread(_preload))
        try:
            yield
        finally:
            stop.set()
            watchdog.cancel()
            if eager is not None and not eager.done():
                eager.cancel()

    app = FastAPI(
        title="MOSS-Image Server (ai-toolkit engine)",
        version=__version__,
        description=(
            "Local MOSS-Image text-to-image inference on ai-toolkit's engine (int8 convrot8, "
            "lazy loading and idle unload. OpenAI-compatible /v1/images/generations "
            "plus native controls."
        ),
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ auth
    async def require_auth(request: Request) -> None:
        expected = config.api_key
        if not expected:
            return
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="invalid API key")

    # --------------------------------------------------------------- helpers
    def build_request(body: ImageGenerationRequest, progress=None):
        return build_generation_request(
            body,
            default_width=config.default_width,
            default_height=config.default_height,
            default_steps=config.default_steps,
            default_guidance=config.default_guidance_scale,
            max_pixels=config.max_pixels,
            default_shift=config.default_shift,
            default_uncond_skip=config.default_uncond_skip_layers,
            progress=progress,
        )

    async def acquire_gpu() -> None:
        try:
            await asyncio.wait_for(app.state.semaphore.acquire(), timeout=config.queue_timeout)
        except asyncio.TimeoutError as error:
            raise HTTPException(
                status_code=503,
                detail="GPU is busy; retry later or raise queue_timeout",
                headers={"Retry-After": "15"},
            ) from error

    async def run_generation(body: ImageGenerationRequest):
        try:
            request, _, _ = build_request(body)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        await acquire_gpu()
        try:
            started = time.perf_counter()
            # Loads the model on the first call, then generates.
            images = await asyncio.to_thread(engine.generate, request)
            return _encode_response(images, body.response_format, config, time.perf_counter() - started)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        finally:
            app.state.semaphore.release()

    # ---------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    async def index():
        return {
            "name": "MOSS-Image Server",
            "version": __version__,
            "state": engine.state,
            "api": {
                "openai": "POST /v1/images/generations",
                "native": "POST /v1/moss/generate",
                "stream": "POST /v1/moss/generate?stream=true",
                "loras": "GET/POST /v1/loras",
                "admin": "POST /v1/admin/load, POST /v1/admin/unload",
                "files": "GET /files/{date}/{file}.png",
            },
        }

    @app.get("/health")
    async def health():
        body = {
            "status": engine.state,
            "version": __version__,
            "model_loaded": engine.ready,
            "idle_unload_seconds": config.idle_unload_seconds,
        }
        if engine.error:
            body["error"] = engine.error
        # "idle" is healthy: the model loads on demand.
        return JSONResponse(body, status_code=503 if engine.state == "error" else 200)

    @app.get("/v1/status")
    async def status():
        info = engine.describe()
        info["version"] = __version__
        info["lazy_load"] = config.lazy_load
        info["queue_timeout"] = config.queue_timeout
        return info

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "moss-image",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "metadata": {
                        "model_dir": config.model_dir,
                        "quantize": config.quantize, "qtype": config.qtype,
                        "state": engine.state,
                        "loras": [entry["name"] for entry in (engine.loras.list() if engine.loras else [])],
                    },
                }
            ],
        }

    @app.post("/v1/images/generations", dependencies=[Depends(require_auth)])
    async def images_generations(body: ImageGenerationRequest):
        return await run_generation(body)

    @app.post("/v1/moss/generate", dependencies=[Depends(require_auth)])
    async def moss_generate(
        body: NativeGenerationRequest,
        stream: bool = Query(False, description="Stream step progress as server-sent events"),
    ):
        if not stream:
            return await run_generation(body)

        try:
            build_request(body)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        await acquire_gpu()

        async def event_stream():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def progress(done: int, total: int) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "step": done, "total": total})

            def worker() -> None:
                try:
                    request, _, _ = build_request(body, progress=progress)
                    started = time.perf_counter()
                    images = engine.generate(request)
                    payload = _encode_response(
                        images, body.response_format, config, time.perf_counter() - started
                    )
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"type": "done", "response": payload.model_dump()}
                    )
                except Exception as error:  # noqa: BLE001 - surfaced to the client
                    LOGGER.exception("streamed generation failed")
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"type": "error", "message": f"{type(error).__name__}: {error}"}
                    )

            thread = threading.Thread(target=worker, name="moss-generate", daemon=True)
            thread.start()
            try:
                yield f"event: status\ndata: {json.dumps({'type': 'status', 'state': engine.state})}\n\n"
                while True:
                    item = await queue.get()
                    yield f"event: {item['type']}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                    if item["type"] in ("done", "error"):
                        break
            finally:
                if thread.is_alive():
                    await asyncio.to_thread(thread.join)
                app.state.semaphore.release()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ LoRA
    def _lora_snapshot() -> dict:
        loaded = engine.loras.list() if engine.loras else []
        return {
            "state": engine.state,
            "loaded": loaded,
            "available": scan_lora_dirs(config.lora_dirs, {entry["name"]: entry["path"] for entry in loaded}),
        }

    @app.get("/v1/loras", dependencies=[Depends(require_auth)])
    async def list_loras():
        return await asyncio.to_thread(_lora_snapshot)

    @app.post("/v1/loras/load", dependencies=[Depends(require_auth)])
    async def load_lora(body: LoRALoadRequest):
        # Registry-only: no model load, so a client can declare its LoRAs up
        # front without paying for (or forcing) a lazy model load.
        def work():
            with engine.lock:
                name = body.name or Path(body.path).stem
                entry = engine.loras.load(name, body.path, scale=body.scale, replace=body.replace)
                return entry.as_dict()

        try:
            return await asyncio.to_thread(work)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/v1/loras/{name}/scale", dependencies=[Depends(require_auth)])
    async def scale_lora(name: str, body: LoRAScaleRequest):
        def work():
            with engine.lock:
                return engine.loras.set_scale(name, body.scale).as_dict()

        try:
            return await asyncio.to_thread(work)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.delete("/v1/loras/{name}", dependencies=[Depends(require_auth)])
    async def unload_lora(name: str):
        def work():
            with engine.lock:
                if engine.loras is None or not engine.loras.unload(name):
                    raise KeyError(name)
                return {"unloaded": name}

        try:
            return await asyncio.to_thread(work)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"LoRA {name!r} is not loaded") from error

    @app.delete("/v1/loras", dependencies=[Depends(require_auth)])
    async def unload_all_loras():
        def work():
            with engine.lock:
                return {"unloaded": engine.loras.clear() if engine.loras else []}

        return await asyncio.to_thread(work)

    # ----------------------------------------------------------------- admin
    @app.post("/v1/admin/load", dependencies=[Depends(require_auth)])
    async def admin_load():
        await acquire_gpu()
        try:
            already = engine.ready
            started = time.perf_counter()
            if not already:
                await asyncio.to_thread(engine.ensure_loaded)
            return {
                "state": engine.state,
                "already_loaded": already,
                "seconds": round(time.perf_counter() - started, 2),
                "load_seconds": round(engine.load_seconds, 2),
            }
        finally:
            app.state.semaphore.release()

    @app.post("/v1/admin/unload", dependencies=[Depends(require_auth)])
    async def admin_unload():
        await acquire_gpu()
        try:
            released = await asyncio.to_thread(engine.unload, "admin request")
            return {"state": engine.state, "unloaded": released, "vram_free_gib": engine.describe().get("gpu_free")}
        finally:
            app.state.semaphore.release()

    @app.post("/v1/warmup", dependencies=[Depends(require_auth)])
    async def warmup():
        await asyncio.to_thread(engine.ensure_loaded)
        seconds = await asyncio.to_thread(engine.warmup)
        return {"warmed_up_in": round(seconds, 2), "state": engine.state}

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=config.output_dir), name="files")
    return app
