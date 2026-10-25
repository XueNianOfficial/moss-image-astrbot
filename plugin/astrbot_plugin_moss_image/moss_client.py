"""HTTP client for the local MOSS-Image server.

Only the standard ``aiohttp`` that ships with AstrBot is required.
"""
from __future__ import annotations

import asyncio
import base64
import urllib.parse
from dataclasses import dataclass, field

import aiohttp


class MossError(Exception):
    """Base error with a user-facing Chinese message."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def user_message(self) -> str:
        return self.message


class MossConnectionError(MossError):
    pass


class MossTimeoutError(MossError):
    pass


class MossServerError(MossError):
    pass


@dataclass
class GeneratedImage:
    data: bytes
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    seconds: float | None = None


@dataclass
class GenerationOutcome:
    images: list[GeneratedImage] = field(default_factory=list)
    seconds: float | None = None
    state_before: str = ""

    @property
    def seed(self) -> int | None:
        return self.images[0].seed if self.images else None


class MossClient:
    """Talks to ``POST /v1/moss/generate`` on the local server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        connect_timeout: float = 10.0,
        generation_timeout: float = 900.0,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:8000").rstrip("/")
        self.api_key = api_key or ""
        self.connect_timeout = connect_timeout
        self.generation_timeout = generation_timeout
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ http
    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(
                        total=self.generation_timeout, connect=self.connect_timeout
                    ),
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                )
            return self._session

    async def close(self) -> None:
        async with self._lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._session = None

    async def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.request(method, url, json=payload) as response:
                text = await response.text()
                if response.status >= 400:
                    raise MossServerError(self._describe_http_error(response.status, text))
                try:
                    return await response.json(content_type=None)
                except Exception as error:  # noqa: BLE001
                    raise MossServerError(f"服务器返回了无法解析的内容：{text[:200]}") from error
        except aiohttp.ClientConnectorError as error:
            raise MossConnectionError(
                f"无法连接 MOSS-Image 服务（{self.base_url}）：{error}"
            ) from error
        except asyncio.TimeoutError as error:
            raise MossTimeoutError(
                f"等待生成超时（{self.generation_timeout:.0f} 秒）。可以调大插件里的超时，或降低分辨率/步数。"
            ) from error
        except aiohttp.ClientError as error:
            raise MossConnectionError(f"与 MOSS-Image 服务通信失败：{error}") from error

    @staticmethod
    def _describe_http_error(status: int, text: str) -> str:
        detail = text.strip()
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, dict) and "detail" in parsed:
                value = parsed["detail"]
                detail = value if isinstance(value, str) else str(value)
        except Exception:  # noqa: BLE001 - keep the raw body
            pass
        if status == 401:
            return "鉴权失败：请检查插件配置中的 API Key 与服务端 --api-key 是否一致"
        if status == 422:
            return f"参数不合法：{detail[:200]}"
        if status == 503:
            return f"服务暂不可用：{detail[:200]}"
        if status >= 500:
            return f"服务端错误（{status}）：{detail[:200]}"
        return f"请求失败（{status}）：{detail[:200]}"

    # ------------------------------------------------------------------- api
    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def status(self) -> dict:
        return await self._request("GET", "/v1/status")

    async def unload(self) -> dict:
        return await self._request("POST", "/v1/admin/unload")

    async def ensure_loaded(self) -> dict:
        return await self._request("POST", "/v1/admin/load")

    # ------------------------------------------------------------------ lora
    async def load_lora(self, name: str, path: str, scale: float = 1.0, *, replace: bool = False) -> dict:
        """Register a LoRA on the server (does not load the model)."""
        return await self._request(
            "POST",
            "/v1/loras/load",
            {"name": name, "path": path, "scale": float(scale), "replace": bool(replace)},
        )

    async def set_lora_scale(self, name: str, scale: float) -> dict:
        quoted = urllib.parse.quote(name, safe="")
        return await self._request("POST", f"/v1/loras/{quoted}/scale", {"scale": float(scale)})

    async def unload_lora(self, name: str) -> dict:
        quoted = urllib.parse.quote(name, safe="")
        return await self._request("DELETE", f"/v1/loras/{quoted}")

    async def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        negative_prompt: str = "",
        seed: int | None = None,
        n: int = 1,
        loras: dict[str, float] | None = None,
        shift: float | None = None,
        uncond_skip_layers: tuple[int, ...] | None = None,
        timeout: float | None = None,
    ) -> GenerationOutcome:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance_scale": float(guidance),
            "n": int(max(1, n)),
            "response_format": "b64_json",
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if loras:
            payload["loras"] = {name: float(value) for name, value in loras.items()}
        if shift is not None:
            payload["shift"] = float(shift)
        if uncond_skip_layers is not None:
            payload["uncond_skip_layers"] = [int(layer) for layer in uncond_skip_layers]

        before = "unknown"
        try:
            health = await self.health()
            before = str(health.get("status", "unknown"))
        except MossError:
            pass

        if timeout:
            session = await self._get_session()
            session.timeout = aiohttp.ClientTimeout(total=timeout, connect=self.connect_timeout)

        data = await self._request("POST", "/v1/moss/generate", payload)
        outcome = GenerationOutcome(seconds=data.get("seconds"), state_before=before)
        for item in data.get("data", []):
            encoded = item.get("b64_json")
            if not encoded:
                continue
            outcome.images.append(
                GeneratedImage(
                    data=base64.b64decode(encoded),
                    seed=item.get("seed"),
                    width=item.get("width"),
                    height=item.get("height"),
                    seconds=item.get("seconds"),
                )
            )
        if not outcome.images:
            raise MossServerError("服务器没有返回图片数据")
        return outcome
