"""AstrBot plugin: MOSS-Image drawing through the local ai-toolkit engine.

Commands
--------
``/draw`` (``/绘图`` ``/画图`` ``/画画`` ``/moss``)
    Draw an image. Supports ``--size``/``--ar``/``--steps``/``--seed``/``--no``/
    ``--lora``/``--enhance``/``--raw``/``--layout``.
``/drawhelp``     usage guide
``/drawstatus``   server state, loaded model, VRAM, idle countdown
``/drawunload``   release the model immediately
``/enhance``      preview the enhanced caption without drawing
``/layout``       preview the compiled Layout prompt without drawing

An LLM tool (``moss_draw_image``) is registered so the bot can decide to draw
on its own during a conversation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star

from .layout_enhancer import LayoutEnhancer, LayoutError
from .lora_config import DEFAULT_CATALOG, LoRACatalog, parse_catalog
from .moss_client import (
    GenerationOutcome,
    MossClient,
    MossConnectionError,
    MossError,
    MossServerError,
    MossTimeoutError,
)
from .prompt_enhancer import PromptEnhancer
from .prompt_parser import DrawRequest, parse_draw_request

TOOL_NAME = "moss_draw_image"
#: Bumped when the handler contract changes; shows up in the load log.
PLUGIN_BUILD = "2-chain-result"

_COMMAND_RE = re.compile(
    r"^[/!#]?\s*(draw|moss|绘图|画图|画画|生图)\b\s*[:：]?\s*", re.IGNORECASE
)
_PREFIX_RE = re.compile(
    r"^(?:(?:请|帮我|帮忙|给我|麻烦|劳驾|拜托|能否|可以|能)\s*)*"
    r"(?:(?:画|绘制|生成|来|做|搞|draw|generate|paint|create|make)\s*(?:一张|一幅|一个|一副|个|张|幅|只|位)?\s*)+"
    r"[:：,，]?\s*",
    re.IGNORECASE,
)


@dataclass
class Settings:
    base_url: str = "http://127.0.0.1:8000"
    api_key: str = ""
    timeout: float = 900.0
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 3.0
    negative_prompt: str = ""
    shift: float | None = None
    uncond_skip_layers: tuple[int, ...] | None = None
    max_images: int = 2
    enhance_enabled: bool = True
    auto_enhance_below: int = 60
    max_prompt_chars: int = 1400
    layout_enabled: bool = True
    layout_retries: int = 2
    lora_enabled: bool = True
    lora_allow_bot_choice: bool = True
    lora_max_weight: float = 1.0
    lora_catalog_raw: object = None
    admin_only: bool = False
    cooldown: float = 10.0
    send_progress: bool = True
    send_parameters: bool = True
    inject_tool_guidance: bool = True


def _section(config, name: str) -> dict:
    try:
        value = config.get(name, {}) or {}
    except Exception:  # noqa: BLE001 - plain dict config
        value = {}
    return value if isinstance(value, dict) else {}


def _bool(mapping: dict, key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_skip_layers(value) -> tuple[int, ...] | None:
    """``"22"`` -> ``(22,)``, ``"22,24"`` -> ``(22, 24)``, empty -> None, "off" -> ()."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("off", "none", "无", "关闭", "不跳过"):
        return ()
    layers: list[int] = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if part:
            layers.append(int(part))
    return tuple(layers) or None


def parse_settings(config) -> Settings:
    server = _section(config, "server")
    generation = _section(config, "generation")
    enhance = _section(config, "enhance")
    layout = _section(config, "layout")
    lora = _section(config, "lora")
    behaviour = _section(config, "behaviour")
    settings = Settings()
    settings.base_url = str(server.get("base_url") or settings.base_url).rstrip("/")
    settings.api_key = str(server.get("api_key") or "")
    settings.timeout = float(server.get("timeout") or settings.timeout)
    settings.width = int(generation.get("width") or settings.width)
    settings.height = int(generation.get("height") or settings.height)
    settings.steps = int(generation.get("steps") or settings.steps)
    settings.guidance = float(generation.get("guidance") or settings.guidance)
    settings.negative_prompt = str(generation.get("negative_prompt") or "").strip()
    shift = generation.get("shift")
    settings.shift = float(shift) if shift not in (None, "") and float(shift) > 0 else None
    settings.uncond_skip_layers = _parse_skip_layers(generation.get("uncond_skip_layers"))
    settings.max_images = max(1, min(4, int(generation.get("max_images") or settings.max_images)))
    settings.enhance_enabled = _bool(enhance, "enabled", settings.enhance_enabled)
    settings.auto_enhance_below = int(enhance.get("auto_enhance_below") or settings.auto_enhance_below)
    settings.max_prompt_chars = int(enhance.get("max_prompt_chars") or settings.max_prompt_chars)
    settings.layout_enabled = _bool(layout, "enabled", settings.layout_enabled)
    settings.layout_retries = int(layout.get("max_retries") or settings.layout_retries)
    settings.lora_enabled = _bool(lora, "enabled", settings.lora_enabled)
    settings.lora_allow_bot_choice = _bool(lora, "allow_bot_choice", settings.lora_allow_bot_choice)
    settings.lora_max_weight = max(0.1, min(2.0, float(lora.get("max_weight") or settings.lora_max_weight)))
    settings.lora_catalog_raw = (
        lora.get("catalog_lines") or lora.get("catalog")
    )
    settings.admin_only = _bool(behaviour, "admin_only", settings.admin_only)
    settings.cooldown = float(behaviour.get("cooldown") or settings.cooldown)
    settings.send_progress = _bool(behaviour, "send_progress", settings.send_progress)
    settings.send_parameters = _bool(behaviour, "send_parameters", settings.send_parameters)
    settings.inject_tool_guidance = _bool(behaviour, "inject_tool_guidance", settings.inject_tool_guidance)
    return settings


class MossImagePlugin(Star):
    """通过本机 MOSS-Image（ai-toolkit 引擎）绘图，带提示词与布局增强。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.settings = parse_settings(config)
        self.client = MossClient(
            self.settings.base_url,
            api_key=self.settings.api_key,
            generation_timeout=self.settings.timeout,
        )
        self.enhancer = PromptEnhancer(
            context,
            enabled=self.settings.enhance_enabled,
            max_chars=self.settings.max_prompt_chars,
        )
        self.layout = LayoutEnhancer(
            context,
            enabled=self.settings.layout_enabled,
            max_retries=self.settings.layout_retries,
        )
        entries = parse_catalog(self.settings.lora_catalog_raw) or parse_catalog(DEFAULT_CATALOG)
        if not self.settings.lora_enabled:
            for entry in entries:
                entry.enabled = False
        self.catalog = LoRACatalog(
            entries,
            max_weight=self.settings.lora_max_weight,
            allow_bot_choice=self.settings.lora_allow_bot_choice,
        )
        self._last_request: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        logger.info(
            f"[moss_image] 已加载（build {PLUGIN_BUILD}），服务地址 {self.settings.base_url}，"
            f"默认 {self.settings.width}x{self.settings.height} / {self.settings.steps} 步"
        )
        try:
            health = await self.client.health()
            logger.info(f"[moss_image] 服务状态: {health.get('status')}")
        except MossError as error:
            logger.warning(f"[moss_image] 暂时无法连接绘图服务: {error.user_message()}")
        await self._register_catalog()

    async def _register_catalog(self) -> None:
        """Declare every catalog LoRA to the server (registry only, no model load)."""
        for entry in self.catalog.entries:
            try:
                await self.client.load_lora(entry.name, entry.path, scale=entry.weight, replace=True)
                entry.registered = True
            except MossError as error:
                entry.registered = False
                logger.warning(f"[moss_image] LoRA {entry.name} 注册失败: {error.user_message()}")
        names = [entry.name for entry in self.catalog.registered()]
        logger.info(
            f"[moss_image] LoRA 目录 {len(self.catalog.entries)} 条，已注册: {names or '无'}"
        )

    async def terminate(self) -> None:
        await self.client.close()

    # ------------------------------------------------------------------ #
    # commands
    # ------------------------------------------------------------------ #
    @filter.command("draw", alias={"绘图", "画图", "画画", "moss", "生图"})
    async def draw(self, event: AstrMessageEvent):
        """使用 MOSS-Image 绘图。用法：/draw 一只戴墨镜的橘猫 --size 1024x1024 --steps 30"""
        if not await self._permitted(event):
            return
        raw = self._strip_command(event.message_str)
        if not raw.strip():
            yield event.plain_result(self._help_text())
            return

        request = parse_draw_request(raw)
        if not request.text:
            yield event.plain_result("🎨 请描述想画的画面\n例如：/draw 雨夜的霓虹街道，一只白狐回头")
            return
        if not await self._check_cooldown(event):
            return

        if self.settings.send_progress:
            yield event.plain_result("🎨 正在生成，请稍候…")

        try:
            components = await self._render(event, request)
        except MossTimeoutError as error:
            yield event.plain_result(f"⏰ {error.user_message()}")
            return
        except MossConnectionError as error:
            yield event.plain_result(self._connection_text(error))
            return
        except MossError as error:
            yield event.plain_result(f"❌ 绘图失败：{error.user_message()}")
            return
        except Exception as error:  # noqa: BLE001 - never break the event loop
            logger.exception(f"[moss_image] 绘图时发生未预期的错误: {error}")
            yield event.plain_result(f"❌ 绘图失败：{error}")
            return

        # The respond stage only consumes MessageEventResult; a bare MessageChain
        # yielded from a handler is dropped without any log.
        yield event.chain_result(components)

    @filter.command("drawhelp", alias={"绘图帮助", "moss帮助"})
    async def draw_help(self, event: AstrMessageEvent):
        """查看绘图插件用法"""
        yield event.plain_result(self._help_text())

    @filter.command("loras", alias={"绘图lora", "lora列表"})
    async def list_loras(self, event: AstrMessageEvent):
        """查看已配置的 LoRA 目录"""
        if not await self._permitted(event):
            return
        lines = ["🧩 LoRA 目录", self.catalog.describe(), ""]
        lines.append(
            "用法：/draw <描述> --lora 名称:权重"
            if self.settings.lora_enabled
            else "LoRA 功能已在配置中关闭"
        )
        if self.catalog.allow_bot_choice:
            lines.append("AI 也会按描述自动选择合适的 LoRA 与权重。")
        yield event.plain_result("\n".join(lines))

    @filter.command("drawstatus", alias={"绘图状态", "moss状态"})
    async def draw_status(self, event: AstrMessageEvent):
        """查看绘图服务状态、模型与显存"""
        if not await self._permitted(event):
            return
        try:
            status = await self.client.status()
        except MossError as error:
            yield event.plain_result(self._connection_text(error))
            return

        state_map = {"ready": "✅ 已加载", "idle": "💤 空闲（未加载）", "loading": "⏳ 加载中", "error": "❌ 出错"}
        state = str(status.get("state", "unknown"))
        lines = [
            "🎨 MOSS-Image 服务",
            f"🔌 地址：{self.settings.base_url}",
            f"📦 状态：{state_map.get(state, state)}",
        ]
        meta = status.get("model_meta") or {}
        if status.get("engine"):
            lines.append(f"⚙️ 引擎：{status['engine']}")
        if status.get("quantize") is not None:
            quant = str(status.get("qtype")) if status.get("quantize") else "bf16"
            lines.append(f"💾 量化：{quant}｜low_vram {status.get('low_vram')}")
        loras = status.get("loras") or []
        if loras:
            lines.append("🧩 LoRA：" + "，".join(f"{item.get('name')}({item.get('scale')})" for item in loras))
        elif meta.get("loras"):
            lines.append("🧩 LoRA：无")
        if state == "ready":
            idle = float(status.get("idle_seconds") or 0)
            limit = int(status.get("idle_unload_seconds") or 0)
            if limit:
                lines.append(f"⏱️ 空闲 {idle:.0f}s，达到 {limit}s 自动卸载")
        used = status.get("vram_used_gib")
        free = status.get("gpu_free")
        if used is not None:
            suffix = f"（剩余 {float(free):.1f} GiB）" if free is not None else ""
            lines.append(f"🖥️ 显存占用：{float(used):.1f} GiB {suffix}")
        elif free is not None:
            lines.append(f"🖥️ 显卡剩余：{float(free):.1f} GiB")
        stats = status.get("stats") or {}
        if stats.get("requests"):
            lines.append(f"📈 已生成 {stats.get('images', 0)} 张 / {stats.get('requests', 0)} 次请求")
        yield event.plain_result("\n".join(lines))

    @filter.command("drawunload", alias={"卸载模型", "释放显存"})
    async def draw_unload(self, event: AstrMessageEvent):
        """立即卸载模型，释放显存"""
        if not await self._permitted(event):
            return
        try:
            result = await self.client.unload()
        except MossError as error:
            yield event.plain_result(self._connection_text(error))
            return
        if result.get("unloaded"):
            free = result.get("vram_free_gib")
            suffix = f"\n🖥️ 显卡剩余：{float(free):.1f} GiB" if free is not None else ""
            yield event.plain_result(f"🧹 已卸载模型，显存已释放{suffix}")
        else:
            yield event.plain_result("ℹ️ 模型当前未加载，无需卸载")

    @filter.command("enhance", alias={"增强提示词", "优化提示词"})
    async def enhance_preview(self, event: AstrMessageEvent):
        """预览提示词增强结果（不绘图）"""
        if not await self._permitted(event):
            return
        raw = self._strip_command(event.message_str)
        if not raw.strip():
            yield event.plain_result("用法：/enhance 一只猫在窗台上")
            return
        enhancement = await self.enhancer.enhance(raw.strip(), umo=event.unified_msg_origin)
        lines = ["✨ 增强后的提示词", enhancement.text]
        if enhancement.warning:
            lines.append(f"⚠️ {enhancement.warning}")
        if not enhancement.used_llm:
            lines.append("（未使用 LLM，原样返回）")
        yield event.plain_result("\n".join(lines))

    @filter.command("layout", alias={"布局增强", "排版提示词"})
    async def layout_preview(self, event: AstrMessageEvent):
        """把描述编译成 MOSS-Image 的 Layout 提示词（不绘图）"""
        if not await self._permitted(event):
            return
        raw = self._strip_command(event.message_str)
        if not raw.strip():
            yield event.plain_result(
                "用法：/layout 一张海报，标题「夏日限定」，右下角一杯冰咖啡 --size 1024x1024"
            )
            return
        request = parse_draw_request(raw)
        width = request.width or self.settings.width
        height = request.height or self.settings.height
        try:
            result = await self.layout.build(
                request.text, width, height, umo=event.unified_msg_origin
            )
        except LayoutError as error:
            yield event.plain_result(f"❌ 布局生成失败：{error}")
            return
        yield event.plain_result(
            f"🧭 Layout 提示词（画布 {result.width}x{result.height}，尝试 {result.attempts} 次）\n"
            f"{result.prompt}\n\n"
            f"用 /draw 同样描述 --layout 即可按此布局出图"
        )

    # ------------------------------------------------------------------ #
    # LLM tool
    # ------------------------------------------------------------------ #
    @filter.llm_tool(name=TOOL_NAME)
    async def draw_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "",
        steps: int = -1,
        seed: int = -1,
        negative_prompt: str = "",
        enhance: bool = True,
        layout: bool = False,
        lora: str = "",
    ):
        """画一张图片并直接发送给用户。当用户要求画图、生成图片、出图、来一张时，必须调用本工具。

        使用场景（命中任意一条就应调用，不要只用文字回应，也不要声称自己不能画图）：
        - 用户要求「画」「画一张」「帮我画」「生成图片」「出图」「来一张」
        - 用户描述了一个画面并希望你把它画出来
        - 用户说「随便画」「你画一个」，即让你自行决定画面内容
        - 用户要求换一张、重画、再来一张，或指定 seed 复现某张图
        - 角色扮演中你想用图片而不是文字来回应某个场景

        生成通常需要数十秒；首次调用还要加载模型（约 30 秒）。
        成功后图片会自动发给用户，你只需一句话简短回应，不要重复描述画面。

        Args:
            prompt(string): 画面描述。中文和英文均可，120字以上效果最佳，最多1200字
            size(string): 画面尺寸，如 1024x1024、1536x1536、16:9。留空使用默认尺寸
            steps(int): 采样步数，-1 使用默认值（50）。
            seed(int): 随机种子，-1 表示随机；复现同一张图时传之前的 seed
            negative_prompt(string): 额外排除的内容，例如 blurry, watermark
            enhance(bool): 是否先把简短描述扩写成详细英文提示词（推荐 true）
            layout(bool): 是否生成带文字/元素定位的 Layout（适合海报、带标题的图）
            lora(string): 可选 LoRA，格式「名称:权重」，多个用逗号，例如 xuenian:1.0。只能使用系统提示里列出的 LoRA；与用户需求不匹配时留空
        """
        if not await self._permitted(event, quiet=True):
            return "绘图权限受限。"

        request = parse_draw_request("")
        subject = self._normalise_prompt_argument(prompt)
        if not subject:
            subject = "a beautiful scenery, highly detailed"
        request.text = subject
        if size:
            parsed = parse_draw_request(f"x --size {size}")
            request.width, request.height = parsed.width, parsed.height
        if steps and int(steps) > 0:
            request.steps = min(200, int(steps))
        if seed is not None and int(seed) >= 0:
            request.seed = int(seed)
        request.negative = (negative_prompt or "").strip()
        request.mode = "layout" if layout else ("caption" if enhance else "raw")
        if lora:
            request.loras = self._parse_lora_argument(lora)

        try:
            components = await self._render(event, request)
        except MossTimeoutError as error:
            return f"生成超时：{error.user_message()}"
        except MossConnectionError as error:
            return self._connection_text(error)
        except MossError as error:
            return f"绘图失败：{error.user_message()}"
        except Exception as error:  # noqa: BLE001
            logger.exception(f"[moss_image] LLM 工具绘图失败: {error}")
            return f"绘图失败：{error}"

        try:
            # event.send() takes a MessageChain (not a MessageEventResult)
            await event.send(MessageChain(components))
        except Exception as error:  # noqa: BLE001
            logger.error(f"[moss_image] 发送图片失败: {error}")
            return f"图片已生成但发送失败：{error}"
        return "图片已生成并发送给用户，请用一句话简短回应，不要重复描述画面。"

    @filter.on_llm_request()
    async def _inject_tool_guidance(self, event: AstrMessageEvent, request) -> None:
        """Remind personas that the drawing tool exists."""
        if not self.settings.inject_tool_guidance:
            return
        available = getattr(request, "func_tool", None)
        if available is None:
            return
        names_method = getattr(available, "names", None)
        if callable(names_method):
            try:
                names = names_method()
            except Exception:  # noqa: BLE001 - unknown tool-set implementation
                names = None
            if names is not None and TOOL_NAME not in names:
                return
        guidance = (
            "【绘图能力】你可以调用 moss_draw_image 工具直接生成图片。"
            "当用户要求画图、出图、生成图片，或描述了一个希望看到的画面时，"
            "请立即调用该工具，不要只用文字回应，也不要声称无法画图。"
        )
        if self.catalog.allow_bot_choice and self.catalog.entries:
            guidance += (
                "\n\n【可用 LoRA】调用 moss_draw_image 时用 lora 参数选择，"
                f"格式「名称:权重」（权重 0~{self.catalog.max_weight:g}），多个用逗号。\n"
                f"{self.catalog.brief()}\n"
                "选择规则：只有当用户的要求与某条描述明显匹配时才启用它，并给出你判断的权重；"
                "不匹配就不要启用，避免风格漂移。「默认启用」的条目已自动生效，无需再次指定。"
            )
        current = getattr(request, "system_prompt", "") or ""
        if guidance in current:
            return
        request.system_prompt = f"{current.rstrip()}\n\n{guidance}".strip()

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    async def _render(self, event: AstrMessageEvent, request: DrawRequest) -> list:
        width = request.width or self.settings.width
        height = request.height or self.settings.height
        steps = request.steps or self.settings.steps
        guidance = request.guidance if request.guidance is not None else self.settings.guidance
        warnings = list(request.warnings)
        prompt = request.text
        used_layout = False
        enhanced = False

        if request.mode == "layout":
            try:
                layout = await self.layout.build(
                    request.text, width, height, umo=event.unified_msg_origin
                )
                prompt, width, height = layout.prompt, layout.width, layout.height
                used_layout = True
            except LayoutError as error:
                warnings.append(f"布局增强失败（{error}），已改用普通描述")

        should_enhance = request.mode == "caption" or (
            request.mode == "auto"
            and self.settings.enhance_enabled
            and len(request.text) < self.settings.auto_enhance_below
        )
        if not used_layout and should_enhance and self.settings.enhance_enabled:
            enhancement = await self.enhancer.enhance(
                request.text, umo=event.unified_msg_origin
            )
            if enhancement.used_llm:
                prompt = enhancement.text
                enhanced = True
            if enhancement.warning:
                warnings.append(enhancement.warning)

        logger.info(
            f"[moss_image] {width}x{height} steps={steps} layout={used_layout} "
            f"enhanced={enhanced} prompt={prompt[:120]!r}"
        )

        # LoRA set + strengths for this request. Disabled entries ride along at
        # weight 0 so ai-toolkit only updates strengths instead of reloading.
        weights, lora_warnings = self.catalog.resolve(request.loras)
        warnings.extend(lora_warnings)
        triggers = self.catalog.trigger_words(weights)
        if triggers:
            prompt = f"{prompt}, {', '.join(triggers)}"
        lora_summary = "，".join(f"{name} {weight:g}" for name, weight in weights.items() if weight > 0)
        negative = ", ".join(part for part in (self.settings.negative_prompt, request.negative) if part)

        try:
            outcome = await self.client.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                negative_prompt=negative,
                seed=request.seed,
                n=min(request.n, self.settings.max_images),
                loras=weights or None,
                shift=self.settings.shift,
                uncond_skip_layers=self.settings.uncond_skip_layers,
            )
        except MossServerError as error:
            # The server keeps LoRAs in memory: after a server restart our
            # registrations are gone. Register again and retry once.
            if "not registered" not in str(error):
                raise
            await self._register_catalog()
            weights, lora_warnings = self.catalog.resolve(request.loras)
            warnings.extend(lora_warnings)
            outcome = await self.client.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                negative_prompt=negative,
                seed=request.seed,
                n=min(request.n, self.settings.max_images),
                loras=weights or None,
                shift=self.settings.shift,
                uncond_skip_layers=self.settings.uncond_skip_layers,
            )
        return self._build_components(
            outcome, width, height, steps, used_layout, enhanced, warnings, lora_summary
        )

    def _build_components(
        self,
        outcome: GenerationOutcome,
        width: int,
        height: int,
        steps: int,
        used_layout: bool,
        enhanced: bool,
        warnings: list[str],
        lora_summary: str = "",
    ) -> list:
        components: list = [Plain("🎨 绘制完成")]
        for generated in outcome.images:
            image = Image.fromBytes(generated.data)
            try:
                image.set_extra("group_id", f"moss_{generated.seed}")
            except Exception:  # noqa: BLE001 - optional on some versions
                pass
            components.append(image)

        if self.settings.send_parameters:
            bits = [f"{width}x{height}", f"{steps} 步"]
            if outcome.seed is not None:
                bits.append(f"seed {outcome.seed}")
            if used_layout:
                bits.append("Layout")
            elif enhanced:
                bits.append("已增强提示词")
            if lora_summary:
                bits.append(f"LoRA {lora_summary}")
            if outcome.seconds:
                bits.append(f"{outcome.seconds:.1f}s")
            components.append(Plain("\n" + "｜".join(bits)))
        if warnings:
            components.append(Plain("\n⚠️ " + "；".join(dict.fromkeys(warnings))))
        return components

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    async def _permitted(self, event: AstrMessageEvent, *, quiet: bool = False) -> bool:
        if not self.settings.admin_only:
            return True
        try:
            if event.is_admin():
                return True
        except Exception:  # noqa: BLE001 - adapter without role info
            logger.debug("[moss_image] 当前平台不支持 is_admin()，拒绝非管理员请求")
        if not quiet:
            await event.send(event.plain_result("🔒 绘图命令仅限管理员使用"))
        return False

    async def _check_cooldown(self, event: AstrMessageEvent) -> bool:
        if self.settings.cooldown <= 0:
            return True
        user = event.get_sender_id()
        now = time.monotonic()
        last = self._last_request.get(user, 0.0)
        if now - last < self.settings.cooldown:
            remaining = self.settings.cooldown - (now - last)
            await event.send(event.plain_result(f"⏳ 冷却中，请 {remaining:.0f} 秒后再试"))
            return False
        self._last_request[user] = now
        return True

    def _connection_text(self, error: MossError) -> str:
        return (
            f"❌ {error.user_message()}\n"
            "请确认 MOSS-Image 服务已启动（装过开机自启会自动拉起，"
            "或手动运行服务端目录下的 run.ps1 / run.bat），"
            "并检查插件配置里的服务地址。"
        )

    def _parse_lora_argument(self, raw: str) -> dict[str, float]:
        """``"xuenian:1.0, kemono:0.5"`` -> ``{"xuenian": 1.0, "kemono": 0.5}``."""
        requested: dict[str, float] = {}
        for part in re.split(r"[,，;；]", raw or ""):
            item = part.strip()
            if not item:
                continue
            name, _, weight = item.partition(":")
            name = name.strip()
            if not name:
                continue
            entry = self.catalog.get(name)
            try:
                value = float(weight) if weight else (entry.weight if entry else 1.0)
            except ValueError:
                value = entry.weight if entry else 1.0
            requested[entry.name if entry else name] = value
        return requested

    @staticmethod
    def _strip_command(message: str) -> str:
        text = (message or "").strip()
        text = _COMMAND_RE.sub("", text, count=1)
        return text.strip()

    @staticmethod
    def _normalise_prompt_argument(raw) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        text = _PREFIX_RE.sub("", text, count=1).strip()
        if len(text) >= 2 and text[0] in "\"'“”「」" and text[-1] in "\"'“”「」":
            text = text[1:-1].strip()
        return text

    def _help_text(self) -> str:
        s = self.settings
        return "\n".join(
            [
                "🎨 MOSS-Image 绘图（ai-toolkit 引擎）",
                "",
                "【基本用法】",
                "/draw <画面描述>",
                "例：/draw 雨夜的霓虹街道，一只白狐回头，电影感",
                "",
                "【尺寸】",
                "  --size 1024x1536   直接指定像素（16 的倍数）",
                "  --ar 16:9          按比例取推荐尺寸",
                "",
                "【参数】",
                f"  --steps 30         采样步数（默认 {s.steps}）",
                f"  --guidance 3       引导强度（默认 {s.guidance}）",
                "  --seed 12345       复现同一张图",
                f"  --n 2              一次出几张（最多 {s.max_images}）",
                "  --no blurry, 水印  负面提示词",
                "",
                "【提示词增强】",
                "  --enhance          强制把描述扩写成详细英文提示词",
                "  --raw              完全按原文出图",
                "  （描述较短时默认自动增强）",
                "",
                "【布局增强】",
                "  --layout           生成带文字/元素定位的 Layout",
                "  例：/draw 一张海报，标题「夏日限定」，右下角一杯冰咖啡 --layout",
                "  /layout <描述>     只预览 Layout 提示词",
                "",
                "【LoRA】",
                "  --lora 名称:权重   指定 LoRA，如 --lora xuenian:1.0",
                "  多个用逗号：--lora kemono:0.5,xuenian:0.9",
                "  /loras            查看已配置的 LoRA 目录",
                "  （AI 会按 LoRA 描述自动决定是否使用及权重）",
                "",
                "【其他】",
                "  /drawstatus   服务状态与显存",
                "  /drawunload   立即卸载模型释放显存",
                "  /enhance <描述>  只预览增强后的提示词",
            ]
        )
