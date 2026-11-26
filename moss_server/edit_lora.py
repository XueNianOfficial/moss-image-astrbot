"""Convert a Moss-Image-Edit adapter into the fused LoRA ai-toolkit expects.

The published adapter (https://huggingface.co/chinoll/Moss-Image-Edit) stores
independent ``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}.{lora_A,lora_B}``
tensors without the ``.weight`` suffix and without per-module alpha (it lives in
``adapter_config.json``).  ai-toolkit's MOSS transformer keeps q/k/v fused as
``proj.qkv``, so the three adapters are stacked:

* down = ``cat([A_q, A_k, A_v])``           (rank -> 3r, shared input)
* up   = ``block_diag(B_q, B_k, B_v)``      (each output slice keeps its own delta)
* alpha -> ``alpha * 3`` so ``alpha/rank`` still equals the trained scale

This is an exact rewrite, not an approximation: the fused linear computes
``[W_q; W_k; W_v] @ x``, so the same block-diagonal delta applies ``B_i A_i`` to
the output slice of each projection.  The conversion runs once and is cached
next to the adapter (see :func:`ensure_fused_adapter`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
#: Only blocks are adapted; these names keep their own alpha.
_PLAIN_SUFFIXES = (".proj.o", ".proj.ffn_up_gate", ".proj.ffn_down")
_FUSED_SUFFIX = ".proj.qkv"


class AdapterError(ValueError):
    """The adapter bundle is missing, malformed or of an unknown format."""


def _read_config(adapter_dir: Path) -> dict:
    path = adapter_dir / ADAPTER_CONFIG
    if not path.is_file():
        raise AdapterError(f"{path} not found")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterError(f"{path} is not valid JSON: {error}") from error
    if config.get("format") != "moss-edit-lora":
        raise AdapterError(
            f"{path}: expected format 'moss-edit-lora', got {config.get('format')!r}"
        )
    modules = config.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise AdapterError(f"{path}: no module metadata")
    return config


def adapter_signature(adapter_dir: Path) -> str:
    """Cheap fingerprint of the adapter bundle (sizes + mtimes)."""
    digest = hashlib.sha256()
    for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS):
        path = adapter_dir / name
        if not path.is_file():
            raise AdapterError(f"{path} not found")
        stat = path.stat()
        digest.update(f"{name}:{stat.st_size}:{int(stat.st_mtime)};".encode())
    return digest.hexdigest()[:16]


def fuse_adapter(adapter_dir: Path, output: Path) -> dict:
    """Write the fused adapter to ``output``; returns a summary dict."""
    from safetensors.torch import load_file, save_file

    config = _read_config(adapter_dir)
    modules = config["modules"]
    weights = adapter_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise AdapterError(f"{weights} not found")

    state = load_file(str(weights))
    fused: dict = {}
    for key, tensor in state.items():
        fused[key + ".weight" if key.endswith((".lora_A", ".lora_B")) else key] = tensor

    entries = modules
    bases = sorted(
        key[: -len(".proj.q.lora_A.weight")]
        for key in fused
        if key.endswith(".proj.q.lora_A.weight")
    )
    if not bases:
        raise AdapterError("no q/k/v LoRA tensors found in the adapter")

    fused_count = 0
    for base in bases:
        names = [f"{base}.proj.{part}." for part in ("q", "k", "v")]
        try:
            downs = [fused.pop(name + "lora_A.weight") for name in names]
            ups = [fused.pop(name + "lora_B.weight") for name in names]
        except KeyError as error:
            raise AdapterError(f"incomplete q/k/v adapter for {base}: {error}") from error
        import torch

        fused[f"{base}{_FUSED_SUFFIX}.lora_A.weight"] = torch.cat(downs, 0).contiguous()
        fused[f"{base}{_FUSED_SUFFIX}.lora_B.weight"] = torch.block_diag(*ups).contiguous()
        spec = entries.get(f"{base}.proj.q", {})
        rank = int(spec.get("rank") or downs[0].shape[0])
        alpha = float(spec.get("alpha") or rank)
        fused[f"{base}{_FUSED_SUFFIX}.alpha"] = torch.tensor(
            alpha * (downs[0].shape[0] * 3 / rank), dtype=torch.float32
        )
        fused_count += 1

    for base, spec in entries.items():
        if base.endswith(_PLAIN_SUFFIXES) and f"{base}.alpha" not in fused:
            fused[f"{base}.alpha"] = torch.tensor(
                float(spec.get("alpha") or spec.get("rank") or 1.0), dtype=torch.float32
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fused, str(output))
    summary = {
        "adapter": str(adapter_dir),
        "fused": str(output),
        "blocks": fused_count,
        "tensors": len(fused),
    }
    LOGGER.info(
        "Fused edit adapter: %d blocks, %d tensors -> %s",
        fused_count, len(fused), output,
    )
    return summary


def ensure_fused_adapter(adapter_dir: Path, cache_dir: Path) -> Path:
    """Fuse once and reuse while the adapter bundle is unchanged."""
    adapter_dir = Path(adapter_dir)
    cache_dir = Path(cache_dir)
    signature = adapter_signature(adapter_dir)
    output = cache_dir / f"moss-edit-lora-{signature}.safetensors"
    marker = output.with_suffix(".json")
    if output.is_file() and marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
        if recorded.get("signature") == signature:
            LOGGER.debug("Reusing fused edit adapter %s", output)
            return output
    summary = fuse_adapter(adapter_dir, output)
    summary["signature"] = signature
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
