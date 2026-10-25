"""LoRA registry for the ai-toolkit engine backend.

The engine (``extensions_built_in/inference_engine``) applies LoRAs per request
from a ``loras`` spec list; in ``hook`` mode it attaches low-rank deltas through
forward hooks, so strengths can change instantly and nothing is merged into the
weights.  This module only remembers name -> (path, strength) and hands the
engine a spec list.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

# Keys any supported LoRA format uses: lora_A/lora_B (peft/diffusers/comfy),
# lora_down/lora_up (kohya), diff (full delta) and lokr_* (LyCORIS).
_LORA_MARKERS = (".lora_A", ".lora_B", ".lora_down", ".lora_up", ".diff", ".lokr_")


@dataclass
class LoRAEntry:
    name: str
    path: str
    scale: float
    keys: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def inspect_lora(path: Path) -> int:
    """Return the number of tensors that look like LoRA weights (header only)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    return sum(1 for key in keys if any(marker in key for marker in _LORA_MARKERS))


class LoRARegistry:
    """Name -> (path, strength) bookkeeping shared by the API and the engine."""

    def __init__(self) -> None:
        self._entries: dict[str, LoRAEntry] = {}

    # -- queries ----------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def get(self, name: str) -> LoRAEntry | None:
        return self._entries.get(name)

    def list(self) -> list[dict]:
        return [entry.as_dict() for entry in self._entries.values()]

    def specs(self, overrides: dict[str, float] | None = None) -> list[dict]:
        """Engine spec list; ``overrides`` temporarily change strengths."""
        overrides = overrides or {}
        specs = []
        for entry in self._entries.values():
            strength = float(overrides.get(entry.name, entry.scale))
            specs.append({"path": entry.path, "strength": strength})
        return specs

    # -- mutation ---------------------------------------------------------
    def load(self, name: str, path: str | Path, scale: float = 1.0, *, replace: bool = False) -> LoRAEntry:
        if name in self._entries and not replace:
            raise ValueError(f"LoRA {name!r} is already loaded (pass replace=true to reload)")
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"only .safetensors LoRA files are supported, got {path.suffix!r}")

        keys = inspect_lora(path)
        if keys == 0:
            raise ValueError(
                f"{path.name}: no LoRA tensors found (expected lora_A/lora_B, lora_down/lora_up, diff or lokr_*)"
            )

        resolved = str(path.resolve())
        # One file, one entry: a second name for the same path would attach the
        # same delta twice (e.g. the server default "RE_MOSS-..._kemono" and a
        # client's catalog name "kemono" for the same .safetensors).
        for other_name, other in list(self._entries.items()):
            if other.path == resolved and other_name != name:
                LOGGER.info("LoRA %s replaced by %s (same file)", other_name, name)
                del self._entries[other_name]

        entry = LoRAEntry(name=name, path=resolved, scale=float(scale), keys=keys)
        self._entries[name] = entry
        LOGGER.info("LoRA registered: %s <- %s (scale %.2f, %d tensors)", name, entry.path, entry.scale, keys)
        return entry

    def unload(self, name: str) -> bool:
        if name not in self._entries:
            return False
        del self._entries[name]
        LOGGER.info("LoRA removed: %s", name)
        return True

    def set_scale(self, name: str, scale: float) -> LoRAEntry:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"LoRA {name!r} is not loaded")
        entry.scale = float(scale)
        return entry

    def clear(self) -> list[str]:
        names = list(self._entries)
        self._entries.clear()
        return names


def scan_lora_dirs(directories: Iterable[str], loaded: dict[str, str] | None = None) -> list[dict]:
    """List ``.safetensors`` candidates under ``directories`` (recursive)."""
    loaded = loaded or {}
    by_path = {str(Path(path).resolve()): name for name, path in loaded.items()}
    found: list[dict] = []
    seen: set[str] = set()
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.safetensors")):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            stat = path.stat()
            found.append(
                {
                    "name": path.stem,
                    "path": resolved,
                    "size_mb": round(stat.st_size / 1024**2, 1),
                    "modified": stat.st_mtime,
                    "loaded_as": by_path.get(resolved),
                }
            )
    found.sort(key=lambda item: item["modified"], reverse=True)
    return found
