"""LoRA catalog: config-declared LoRAs with descriptions the LLM can reason about.

Each entry is a small record (name, path, description, enabled, weight, trigger
words).  The catalog is the single source of truth for the plugin:

* every entry is registered on the MOSS server at startup (disabled ones with
  weight 0, which ai-toolkit's hooks short-circuit);
* each request sends an explicit ``{name: weight}`` map, so nothing drifts;
* the LLM gets a plain-text brief of the catalog so it can decide which LoRA
  (if any) matches the user's request and at what weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Used when the user has not configured a catalog yet.
DEFAULT_CATALOG: list[dict] = []


@dataclass
class LoRAEntry:
    name: str
    path: str
    description: str = ""
    enabled: bool = True
    weight: float = 0.7
    trigger_words: str = ""
    #: Filled in by the plugin after a successful /v1/loras/load
    registered: bool = field(default=False, compare=False)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "trigger_words": self.trigger_words,
            "registered": self.registered,
        }


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on", "启用")


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_catalog_text(text: str) -> list[dict]:
    """Parse the editable one-line-per-LoRA format.

    ``name | path | on/off | weight | description | trigger_words``
    Entries are separated by newlines or ``;``; ``#`` starts a comment.
    """
    entries: list[dict] = []
    for chunk in str(text or "").replace("\r", "").split(";"):
        for line in chunk.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 2:
                continue
            item = {"name": parts[0], "path": parts[1]}
            if len(parts) > 2 and parts[2]:
                item["enabled"] = parts[2].lower() not in ("off", "false", "0", "no", "关闭", "禁用")
            if len(parts) > 3 and parts[3]:
                item["weight"] = parts[3]
            if len(parts) > 4:
                item["description"] = parts[4]
            if len(parts) > 5:
                item["trigger_words"] = parts[5]
            entries.append(item)
    return entries


def parse_catalog(raw) -> list[LoRAEntry]:
    """Accept the text DSL, a list of dicts, a dict keyed by name, or paths."""
    entries: list[LoRAEntry] = []
    seen: set[str] = set()

    def add(item) -> None:
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            return
        path = str(item.get("path") or "").strip()
        name = str(item.get("name") or "").strip() or (Path(path).stem if path else "")
        if not path or not name or name in seen:
            return
        seen.add(name)
        entries.append(
            LoRAEntry(
                name=name,
                path=path,
                description=str(item.get("description") or "").strip(),
                enabled=_as_bool(item.get("enabled"), True),
                weight=max(0.0, _as_float(item.get("weight"), 0.7)),
                trigger_words=str(item.get("trigger_words") or "").strip(),
            )
        )

    if isinstance(raw, str):
        text = raw.strip()
        if text and "|" not in text and ("/" in text or "\\" in text):
            # a single bare path is still accepted
            add(text)
        else:
            for item in parse_catalog_text(text):
                add(item)
    elif isinstance(raw, dict):
        for name, value in raw.items():
            if isinstance(value, dict):
                add({**value, "name": value.get("name") or name})
            elif isinstance(value, str):
                add({"name": name, "path": value})
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            add(item)
    return entries


class LoRACatalog:
    def __init__(self, entries: list[LoRAEntry], *, max_weight: float = 1.0, allow_bot_choice: bool = True):
        self.entries = entries
        self.max_weight = max(0.1, max_weight)
        self.allow_bot_choice = allow_bot_choice

    # -- lookups ----------------------------------------------------------
    def get(self, name: str) -> LoRAEntry | None:
        wanted = (name or "").strip().lower()
        for entry in self.entries:
            if entry.name.lower() == wanted:
                return entry
        return None

    def enabled(self) -> list[LoRAEntry]:
        return [entry for entry in self.entries if entry.enabled]

    def registered(self) -> list[LoRAEntry]:
        return [entry for entry in self.entries if entry.registered]

    # -- weights ----------------------------------------------------------
    def resolve(self, requested: dict[str, float] | None = None) -> tuple[dict[str, float], list[str]]:
        """Full ``{name: weight}`` map for one request, plus warnings.

        Disabled entries are sent as 0.0 so the server keeps a stable LoRA set
        (ai-toolkit then only updates strengths, never reloads the files) and
        the hooks short-circuit at zero.
        """
        warnings: list[str] = []
        weights: dict[str, float] = {}
        for entry in self.registered():
            weights[entry.name] = entry.weight if entry.enabled else 0.0

        for name, value in (requested or {}).items():
            entry = self.get(name)
            if entry is None:
                warnings.append(f"未知 LoRA：{name}")
                continue
            if not entry.registered:
                warnings.append(f"LoRA {name} 未注册到服务端（文件可能不存在）")
                continue
            weight = max(0.0, min(self.max_weight, _as_float(value, entry.weight)))
            weights[entry.name] = weight
        return weights, warnings

    def trigger_words(self, weights: dict[str, float]) -> list[str]:
        words: list[str] = []
        for entry in self.entries:
            if weights.get(entry.name, 0.0) > 0 and entry.trigger_words:
                for part in entry.trigger_words.replace("，", ",").split(","):
                    part = part.strip()
                    if part and part not in words:
                        words.append(part)
        return words

    # -- text for the LLM / humans ---------------------------------------
    def brief(self) -> str:
        """Compact catalog description injected into the system prompt."""
        if not self.entries:
            return ""
        lines = []
        for entry in self.entries:
            state = "默认启用" if entry.enabled else "默认关闭"
            if not entry.registered:
                state = "不可用（文件缺失）"
            desc = entry.description or "（无描述）"
            line = f"- {entry.name}（默认权重 {entry.weight:g}，{state}）：{desc}"
            if entry.trigger_words:
                line += f" 触发词：{entry.trigger_words}"
            lines.append(line)
        return "\n".join(lines)

    def describe(self) -> str:
        """Human-facing list used by /loras."""
        if not self.entries:
            return "（未配置任何 LoRA）"
        lines = []
        for entry in self.entries:
            flag = "✅" if entry.enabled else "▫️"
            if not entry.registered:
                flag = "⚠️"
            line = f"{flag} {entry.name}｜权重 {entry.weight:g}"
            if entry.description:
                line += f"｜{entry.description}"
            if not entry.registered:
                line += "｜（文件缺失，未注册）"
            lines.append(line)
        return "\n".join(lines)
