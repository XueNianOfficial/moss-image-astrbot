"""Editing tests: request validation, adapter fusion and lifecycle.

No GPU and no model load: ``MossEditEngine`` is only constructed, never
started, so these run on any machine and in CI.  The fusion test builds a tiny
synthetic adapter in Moss-Image-Edit's layout and checks the exact block form.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from moss_server.config import load_config  # noqa: E402
from moss_server.edit import EditError, EditService  # noqa: E402
from moss_server.edit_lora import (  # noqa: E402
    AdapterError,
    adapter_signature,
    ensure_fused_adapter,
    fuse_adapter,
)
from moss_server.schemas import (  # noqa: E402
    MossEditRequest,
    build_edit_request,
    decode_edit_images,
)


def png_bytes(width: int = 640, height: int = 960, colour=(30, 40, 50)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def as_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def build(body: MossEditRequest, **overrides):
    kwargs = dict(
        default_width=1024,
        default_height=1024,
        max_pixels=2048 * 2048,
        default_steps=20,
        default_guidance=4.0,
        default_shift=5.0,
        max_ref_images=4,
        default_max_edge=1024,
    )
    kwargs.update(overrides)
    return build_edit_request(body, **kwargs)


def write_adapter(root: Path, *, layers=(0, 1), rank: int = 2, dim: int = 8, ffn: int = 16) -> Path:
    """A minimal adapter in the published moss-edit-lora layout."""
    adapter = root / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    modules = {}
    tensors = {}
    for layer in layers:
        for name, in_features, out_features in (
            ("q", dim, dim),
            ("k", dim, dim),
            ("v", dim, dim),
            ("o", dim, dim),
            ("ffn_up_gate", dim, 2 * ffn),
            ("ffn_down", ffn, dim),
        ):
            base = f"layers.{layer}.proj.{name}"
            modules[base] = {
                "rank": rank,
                "alpha": float(rank),
                "dropout": 0.0,
                "in_features": in_features,
                "out_features": out_features,
                "param_dtype": "bfloat16",
            }
            tensors[f"{base}.lora_A"] = torch.randn(rank, in_features, dtype=torch.bfloat16)
            tensors[f"{base}.lora_B"] = torch.randn(out_features, rank, dtype=torch.bfloat16)
    config = {
        "format": "moss-edit-lora",
        "format_version": 1,
        "targets": sorted(modules),
        "modules": modules,
        "metadata": {"dit_quantization": "none"},
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    from safetensors.torch import save_file

    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    return adapter


def test_decode_accepts_base64_and_data_url():
    raw = png_bytes()
    images = decode_edit_images([as_b64(raw), f"data:image/png;base64,{as_b64(raw)}"])
    assert len(images) == 2 and images[0] == raw
    print(f"  decoded {len(images)} images")


def test_decode_rejects_bad_input():
    for payloads, label in (([], "empty"), (["  "], "blank"), (["not-base64!!"], "bad base64")):
        try:
            decode_edit_images(payloads)
        except ValueError:
            print(f"  rejected: {label}")
            continue
        raise AssertionError(f"{label} should have been rejected")


def test_canvas_follows_reference_aspect():
    raw = png_bytes(640, 960)
    _, width, height = build(MossEditRequest(prompt="x", images=[as_b64(raw)]))
    assert width % 16 == 0 and height % 16 == 0 and height > width
    assert 0.6 < (width / height) / (640 / 960) < 1.4
    print(f"  derived canvas: {width}x{height}")


def test_explicit_size_wins():
    raw = png_bytes()
    _, width, height = build(MossEditRequest(prompt="x", images=[as_b64(raw)], size="768x512"))
    assert (width, height) == (768, 512)
    for bad in ("768x500", "12x12", "4096x4096"):
        try:
            build(MossEditRequest(prompt="x", images=[as_b64(raw)], size=bad))
        except ValueError:
            continue
        raise AssertionError(f"size {bad} should have been rejected")
    print("  explicit size honoured, invalid rejected")


def test_defaults_and_overrides():
    raw = png_bytes()
    request, _, _ = build(MossEditRequest(prompt="x", images=[as_b64(raw)]))
    assert request.steps == 20 and request.guidance_scale == 4.0 and request.shift == 5.0
    tuned, _, _ = build(
        MossEditRequest(
            prompt="x", images=[as_b64(raw), as_b64(raw)],
            steps=8, guidance_scale=3.5, shift=3.0, seed=77, max_edge=512,
        )
    )
    assert tuned.steps == 8 and tuned.guidance_scale == 3.5 and tuned.seed == 77
    assert tuned.max_edge == 512 and len(tuned.images) == 2
    print("  defaults + per-request overrides")


def test_fusion_is_exact():
    root = Path(tempfile.mkdtemp(prefix="moss-edit-lora-"))
    adapter = write_adapter(root)
    output = root / "fused.safetensors"
    summary = fuse_adapter(adapter, output)
    assert summary["blocks"] == 2, summary

    from safetensors.torch import load_file

    fused = load_file(str(output))
    for layer in (0, 1):
        base = f"layers.{layer}.proj"
        a_qkv = fused[f"{base}.qkv.lora_A.weight"]
        b_qkv = fused[f"{base}.qkv.lora_B.weight"]
        assert a_qkv.shape == (6, 8), a_qkv.shape          # rank 2 x 3 projections
        assert b_qkv.shape == (24, 6), b_qkv.shape         # 3 * dim rows
        assert float(fused[f"{base}.qkv.alpha"]) == 6.0    # alpha 2 * 3
        # block diagonal: the k slice only sees k rows, so the fused delta must
        # equal the three separate deltas on their own output slices.
        x = torch.randn(4, 8).to(a_qkv.dtype)
        delta = ((x @ a_qkv.T) @ b_qkv.T).float()
        for index, name in enumerate(("q", "k", "v")):
            a = fused[f"{base}.{name}.lora_A.weight"] if f"{base}.{name}.lora_A.weight" in fused else None
            assert a is None, "q/k/v adapters must be folded into qkv"
            assert delta[:, index * 8 : (index + 1) * 8].abs().sum() > 0
    assert fused[f"{base}.o.alpha"].item() == 2.0
    print("  fused adapter: exact block form, alpha scaled by 3")


def test_fusion_cache_reuses_and_invalidates():
    root = Path(tempfile.mkdtemp(prefix="moss-edit-cache-"))
    adapter = write_adapter(root)
    cache = root / "cache"
    first = ensure_fused_adapter(adapter, cache)
    assert first.is_file()
    marker = first.with_suffix(".json")
    assert json.loads(marker.read_text(encoding="utf-8"))["signature"] == adapter_signature(adapter)
    stamp = first.stat().st_mtime_ns
    again = ensure_fused_adapter(adapter, cache)
    assert again == first and again.stat().st_mtime_ns == stamp, "cache should be reused"
    # touching the adapter changes the signature -> a new fused file
    (adapter / "adapter_config.json").write_text(
        (adapter / "adapter_config.json").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    rebuilt = ensure_fused_adapter(adapter, cache)
    assert rebuilt != first, "changed adapter must produce a new fused file"
    print("  fusion cache reused, then invalidated on change")


def test_bad_adapter_is_reported():
    root = Path(tempfile.mkdtemp(prefix="moss-edit-bad-"))
    adapter = root / "adapter"
    adapter.mkdir()
    try:
        adapter_signature(adapter)
    except AdapterError:
        print("  missing files rejected")
    (adapter / "adapter_config.json").write_text('{"format": "other"}', encoding="utf-8")
    try:
        adapter_signature(adapter)
    except AdapterError:
        pass
    try:
        fuse_adapter(adapter, root / "out.safetensors")
    except AdapterError:
        print("  wrong format rejected")


def test_service_unavailable_without_adapter():
    config = load_config(None, None)
    config.edit_adapter = ""
    service = EditService(config)
    available, reason = service.availability()
    assert not available and "edit_adapter" in reason
    assert service.status()["state"] == "idle"
    try:
        service.edit(None)
    except EditError as error:
        assert "edit_adapter" in str(error)
    print(f"  unavailable as expected: {reason}")


def test_service_availability_and_idle_stop():
    root = Path(tempfile.mkdtemp(prefix="moss-edit-svc-"))
    adapter = write_adapter(root)
    config = load_config(None, None)
    config.output_dir = str(root / "out")
    config.edit_adapter = str(adapter)
    service = EditService(config)
    available, reason = service.availability()
    assert available, reason
    status = service.status()
    assert status["available"] and status["state"] == "idle" and status["backed_by"] == "ai-toolkit"
    assert service.maybe_idle_stop() is False
    assert service.stop() is False
    print("  availability + idle-stop no-op verified")


def test_config_edit_fields():
    config = load_config(None, None)
    assert config.edit_enabled is True and config.edit_steps == 20
    assert config.edit_uncond_skip_layers == []
    assert config.edit_free_t2i is True

    root = Path(tempfile.mkdtemp(prefix="moss-edit-cfg-"))
    (root / "ai-toolkit" / "toolkit").mkdir(parents=True)
    (root / "model" / "transformer").mkdir(parents=True)
    base = {"aitk_root": str(root / "ai-toolkit"), "model_dir": str(root / "model")}
    assert load_config(None, base).validate() is not None
    for field, value in (("edit_max_ref_images", 99), ("edit_max_edge", 8)):
        try:
            load_config(None, {**base, field: value}).validate()
        except ValueError as error:
            assert field in str(error), error
            continue
        raise AssertionError(f"{field}={value} should fail validation")
    print("  edit config validates ranges")


def _run() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        print(f"[{test.__name__}]")
        try:
            test()
            print("  ok")
        except AssertionError as error:
            failures += 1
            print(f"  FAILED: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
