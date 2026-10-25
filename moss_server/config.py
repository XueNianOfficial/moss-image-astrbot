"""Server configuration: dataclass defaults, TOML file, CLI overrides, env vars.

Precedence (low to high): dataclass defaults -> TOML file -> environment
variables -> CLI flags.

The model is served through **ai-toolkit's inference engine**
(``extensions_built_in/inference_engine``) with its convrot8 quantisation,
layer offloading and hook-mode LoRA loading, so weights are never merged.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEVICE_CHOICES = ("cuda", "cpu", "auto")
LORA_MODES = ("hook", "merge")


@dataclass
class Config:
    # --- ai-toolkit -------------------------------------------------------
    # The checkout that provides the inference engine and the MOSS-Image model.
    # install.ps1 / install.sh fill these in config.toml.
    aitk_root: str = ""
    # Base diffusers bundle (transformer/, text_encoder/, tokenizer/, vae/).
    # No merged copy is used: LoRA is attached at runtime instead.
    model_dir: str = ""
    device: str = "cuda"
    dtype: str = "bf16"
    # Optional explicit VAE file; empty lets ai-toolkit find
    # <model_dir>/vae/diffusion_pytorch_model.safetensors.
    vae_path: str = ""

    # --- memory / quantisation -------------------------------------------
    # convrot8 = ai-toolkit's int8 weight quantisation (what its training used).
    # Set quantize = false for a full bf16 model (~20 GiB of weights).
    quantize: bool = True
    qtype: str = "convrot8"
    quantize_te: bool = True
    qtype_te: str = "convrot8"
    # low_vram keeps components on the CPU until a request needs them.
    low_vram: bool = True
    layer_offloading: bool = True
    # Fraction of each component's layers streamed from RAM during the forward
    # pass. 0 keeps the transformer resident once used; 1 offloads everything.
    layer_offloading_transformer_percent: float = 0.0
    layer_offloading_text_encoder_percent: float = 1.0

    # --- lifecycle --------------------------------------------------------
    lazy_load: bool = True
    idle_unload_seconds: int = 600
    warmup: bool = True
    generation_timeout: float = 900.0
    queue_timeout: float = 900.0

    # --- generation defaults ---------------------------------------------
    default_steps: int = 30
    default_guidance_scale: float = 3.0
    default_height: int = 1024
    default_width: int = 1024
    default_max_sequence_length: int = 4096
    # MOSS-Image skips this DiT block in the unconditional branch (CFG).
    default_uncond_skip_layers: list[int] = field(default_factory=lambda: [22])
    # Flow-matching sigma shift (the vendor default is 5.0).
    default_shift: float = 5.0
    max_pixels: int = 2048 * 2048

    # --- LoRA -------------------------------------------------------------
    # Files attached at startup (diffusers / ai-toolkit / kohya are all parsed
    # by ai-toolkit's InferenceLoRA). "hook" applies low-rank deltas through
    # forward hooks: strengths change instantly and nothing is merged.
    loras: list[str] = field(default_factory=list)
    lora_scale: float = 0.7
    lora_mode: str = "hook"
    lora_dirs: list[str] = field(default_factory=list)

    # --- HTTP -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""
    public_base_url: str = ""
    output_dir: str = "outputs"
    log_level: str = "info"

    def resolve(self, base: Path) -> "Config":
        """Resolve relative paths against ``base`` and normalise values."""
        if not Path(self.aitk_root).is_absolute():
            self.aitk_root = str((base / self.aitk_root).resolve())
        if not Path(self.model_dir).is_absolute():
            self.model_dir = str((base / self.model_dir).resolve())
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str((base / self.output_dir).resolve())
        if self.vae_path:
            self.vae_path = str(Path(self.vae_path).expanduser().resolve())
        self.loras = [str(Path(p).expanduser().resolve()) for p in self.loras if p]
        self.lora_dirs = [str(Path(p).expanduser().resolve()) for p in self.lora_dirs]
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.public_base_url:
            self.public_base_url = f"http://{self.host}:{self.port}"
        self.public_base_url = self.public_base_url.rstrip("/")
        return self

    def validate(self) -> "Config":
        if self.lora_mode not in LORA_MODES:
            raise ValueError(f"lora_mode must be one of {LORA_MODES}, got {self.lora_mode!r}")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be cuda or cpu, got {self.device!r}")
        for key in ("layer_offloading_transformer_percent", "layer_offloading_text_encoder_percent"):
            value = getattr(self, key)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be within 0..1, got {value}")
        if not 1 <= self.default_steps <= 200:
            raise ValueError("default_steps must be within 1..200")
        if not Path(self.aitk_root, "toolkit").is_dir():
            raise ValueError(f"aitk_root does not look like an ai-toolkit checkout: {self.aitk_root}")
        if not Path(self.model_dir, "transformer").is_dir():
            raise ValueError(f"model_dir does not contain transformer/: {self.model_dir}")
        return self


_FIELDS = {f.name: f for f in dataclasses.fields(Config)}


def _coerce(name: str, value: Any) -> Any:
    """Coerce a TOML/env string into the dataclass field type."""
    field_type = _FIELDS[name].type
    target = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", "str")
    if target.startswith("list["):
        inner = target[len("list["):-1]
        items = value if not isinstance(value, str) else [
            item for item in value.replace(";", ",").split(",") if item.strip()
        ]
        if inner == "int":
            return [int(item) for item in items]
        if inner == "float":
            return [float(item) for item in items]
        return [str(item).strip() for item in items]
    if target == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if target == "int":
        return int(value)
    if target == "float":
        return float(value)
    return str(value)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config from a TOML file, env vars and an override mapping."""
    data: dict[str, Any] = {}
    if path:
        with open(path, "rb") as handle:
            document = tomllib.load(handle)
        # Accept both a flat document and one nested under [server].
        data.update(document.get("server", document))

    env_prefix = "MOSS_"
    for name in _FIELDS:
        env_value = os.environ.get(env_prefix + name.upper())
        if env_value is not None:
            data[name] = env_value

    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    clean = {key: _coerce(key, value) for key, value in data.items() if key in _FIELDS}
    return Config(**clean)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moss-image-server",
        description="MOSS-Image API server powered by ai-toolkit's inference engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="TOML configuration file")
    parser.add_argument("--aitk-root", default=None, help="ai-toolkit checkout")
    parser.add_argument("--model-dir", default=None, help="MOSS-Image diffusers bundle (bf16)")
    parser.add_argument("--device", default=None, choices=DEVICE_CHOICES)
    parser.add_argument("--dtype", default=None, choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--vae-path", default=None, help="explicit VAE safetensors (optional)")

    parser.add_argument("--quantize", dest="quantize", action="store_true", default=None)
    parser.add_argument("--no-quantize", dest="quantize", action="store_false", default=None)
    parser.add_argument("--qtype", default=None)
    parser.add_argument("--quantize-te", dest="quantize_te", action="store_true", default=None)
    parser.add_argument("--no-quantize-te", dest="quantize_te", action="store_false", default=None)
    parser.add_argument("--qtype-te", default=None)
    parser.add_argument("--low-vram", dest="low_vram", action="store_true", default=None)
    parser.add_argument("--no-low-vram", dest="low_vram", action="store_false", default=None)
    parser.add_argument("--layer-offloading", dest="layer_offloading", action="store_true", default=None)
    parser.add_argument("--no-layer-offloading", dest="layer_offloading", action="store_false", default=None)
    parser.add_argument("--offload-transformer", type=float, default=None,
                        dest="layer_offloading_transformer_percent")
    parser.add_argument("--offload-text-encoder", type=float, default=None,
                        dest="layer_offloading_text_encoder_percent")

    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--public-base-url", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--queue-timeout", type=float, default=None)
    parser.add_argument("--generation-timeout", type=float, default=None)
    parser.add_argument("--idle-unload", type=int, default=None, dest="idle_unload_seconds",
                        help="seconds of inactivity before the model is unloaded (0 = never)")
    parser.add_argument("--eager-load", action="store_true", default=None,
                        help="load the model at startup instead of on the first request")

    parser.add_argument("--lora", action="append", default=None, dest="loras",
                        help="LoRA safetensors attached at startup (repeatable)")
    parser.add_argument("--lora-dir", "--lora-dirs", action="append", default=None, dest="lora_dirs",
                        help="directory scanned by GET /v1/loras (repeatable)")
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--lora-mode", default=None, choices=LORA_MODES,
                        help="hook = dynamic deltas (no merge), merge = fold into weights")

    parser.add_argument("--steps", type=int, default=None, dest="default_steps")
    parser.add_argument("--guidance-scale", type=float, default=None, dest="default_guidance_scale")
    parser.add_argument("--shift", type=float, default=None, dest="default_shift",
                        help="flow-matching sigma shift (vendor default 5.0)")
    parser.add_argument("--uncond-skip-layers", type=int, nargs="*", default=None,
                        dest="default_uncond_skip_layers",
                        help="DiT blocks skipped in the unconditional branch")
    parser.add_argument("--log-level", default=None, choices=("debug", "info", "warning", "error"))
    parser.add_argument("--no-warmup", action="store_true", default=None)
    return parser


def config_from_args(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    if overrides.pop("no_warmup", None):
        overrides["warmup"] = False
    if overrides.pop("eager_load", None):
        overrides["lazy_load"] = False
    cfg = load_config(args.config, overrides)
    return cfg.resolve(Path.cwd()).validate(), args
