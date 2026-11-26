"""Editing request/schema tests (no GPU, no model load).

The heavy path (worker startup, sampling) is exercised by
``tests/test_plugin_integration.py`` with ``MOSS_TEST_EDIT=1`` against a running
server; here we only cover the validation and lifecycle logic that must work on
any machine.
"""
from __future__ import annotations

import base64
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from moss_server.config import load_config  # noqa: E402
from moss_server.edit import EditService  # noqa: E402
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


def test_decode_accepts_base64_and_data_url():
    raw = png_bytes()
    images = decode_edit_images([as_b64(raw), f"data:image/png;base64,{as_b64(raw)}"])
    assert len(images) == 2
    assert images[0] == raw and images[1] == raw
    print(f"  decoded {len(images)} images ({len(raw)} bytes each)")


def test_decode_rejects_bad_input():
    for payloads, label in (
        ([], "empty list"),
        (["   "], "blank entry"),
        (["not-base64!!"], "invalid base64"),
        (["aGVsbG8="], "unrelated bytes"),  # decodes, but is not an image
    ):
        try:
            images = decode_edit_images(payloads)
        except ValueError:
            print(f"  rejected: {label}")
            continue
        assert label == "unrelated bytes", f"{label} should have been rejected"
        assert images[0] == b"hello"
    try:
        decode_edit_images(["aGVsbG8="], max_images=0)
    except ValueError:
        print("  rejected: too many images")


def test_canvas_follows_reference_aspect():
    raw = png_bytes(640, 960)  # 2:3 portrait
    request, width, height = build(MossEditRequest(prompt="x", images=[as_b64(raw)]))
    assert width % 16 == 0 and height % 16 == 0
    assert height > width, (width, height)
    assert 0.6 < (width / height) / (640 / 960) < 1.4, (width, height)
    assert width * height <= 2048 * 2048
    print(f"  derived canvas: {width}x{height} from 640x960")


def test_explicit_size_wins():
    raw = png_bytes()
    request, width, height = build(MossEditRequest(prompt="x", images=[as_b64(raw)], size="768x512"))
    assert (width, height) == (768, 512)
    request, width, height = build(MossEditRequest(prompt="x", images=[as_b64(raw)], width=896, height=896))
    assert (width, height) == (896, 896)
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
    assert request.use_kv_cache is True
    tuned, _, _ = build(
        MossEditRequest(
            prompt="x",
            images=[as_b64(raw)],
            steps=8,
            guidance_scale=3.5,
            shift=3.0,
            seed=77,
            max_edge=512,
            use_kv_cache=False,
        )
    )
    assert tuned.steps == 8 and tuned.guidance_scale == 3.5 and tuned.shift == 3.0
    assert tuned.seed == 77 and tuned.max_edge == 512 and tuned.use_kv_cache is False
    print("  defaults + per-request overrides")


def test_body_validation():
    raw = png_bytes()
    try:
        MossEditRequest(prompt="", images=[as_b64(raw)])
    except Exception:
        print("  empty prompt rejected")
    try:
        MossEditRequest(prompt="x", images=[as_b64(raw)], unknown_field=1)
    except Exception:
        print("  unknown field rejected")


def test_service_unavailable_without_paths():
    config = load_config(None, None)
    config.edit_repo = ""
    config.edit_adapter = ""
    service = EditService(config)
    available, reason = service.availability()
    assert not available and "edit_repo" in reason
    assert service.status()["state"] == "idle"
    try:
        service.edit(None)
    except Exception as error:
        assert "edit_repo" in str(error), error
    print(f"  unavailable as expected: {reason}")


def test_service_availability_and_idle_stop():
    root = Path(tempfile.mkdtemp(prefix="moss-edit-"))
    repo = root / "Moss-Image-Edit"
    (repo / "moss_edit").mkdir(parents=True)
    (repo / "moss_edit" / "model.py").write_text("# stub\n", encoding="utf-8")
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"stub")

    config = load_config(None, None)
    config.output_dir = str(root)
    config.edit_repo = str(repo)
    config.edit_adapter = str(adapter)
    service = EditService(config)
    available, reason = service.availability()
    assert available, reason
    status = service.status()
    assert status["available"] and status["state"] == "idle"

    # Not running -> idle stop is a no-op, and a request fails cleanly.
    assert service.maybe_idle_stop() is False
    assert service.stop() is False
    print("  availability + idle-stop no-op verified")


def test_config_edit_fields():
    config = load_config(None, None)
    assert config.edit_enabled is True
    assert config.edit_steps == 20 and config.edit_guidance_scale == 4.0
    assert config.edit_max_ref_images == 4 and config.edit_max_edge == 1024
    assert config.edit_free_t2i is True

    # Range checks need valid paths first, so build a throwaway layout.
    root = Path(tempfile.mkdtemp(prefix="moss-edit-cfg-"))
    (root / "ai-toolkit" / "toolkit").mkdir(parents=True)
    (root / "model" / "transformer").mkdir(parents=True)
    base = {
        "aitk_root": str(root / "ai-toolkit"),
        "model_dir": str(root / "model"),
    }
    assert load_config(None, base).validate() is not None

    for field, value, expected in (
        ("edit_max_ref_images", 99, "edit_max_ref_images"),
        ("edit_max_edge", 8, "edit_max_edge"),
    ):
        try:
            load_config(None, {**base, field: value}).validate()
        except ValueError as error:
            assert expected in str(error), error
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
