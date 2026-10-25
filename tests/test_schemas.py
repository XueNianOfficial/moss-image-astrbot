"""Request parsing and validation (no model needed)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_server.config import Config, load_config
from moss_server.schemas import ImageGenerationRequest, build_generation_request


def build(body: dict, **kwargs):
    request = ImageGenerationRequest(**body)
    return build_generation_request(
        request,
        default_width=1024,
        default_height=1024,
        default_steps=30,
        default_guidance=3.0,
        max_pixels=2048 * 2048,
        **kwargs,
    )


def test_size_string():
    request, width, height = build({"prompt": "cat", "size": "768x512"})
    assert (width, height) == (768, 512)
    assert (request.width, request.height) == (768, 512)


def test_size_object_and_defaults():
    _, width, height = build({"prompt": "cat", "size": {"width": 640, "height": 896}})
    assert (width, height) == (640, 896)

    request, width, height = build({"prompt": "cat"})
    assert (width, height) == (1024, 1024)
    assert request.steps == 30
    assert request.guidance_scale == 3.0
    assert request.num_images == 1


def test_invalid_sizes_rejected():
    for size in ("1000x1000", "1024", "48x48", "4096x4096"):
        try:
            build({"prompt": "cat", "size": size})
        except ValueError:
            continue
        raise AssertionError(f"size {size!r} should be rejected")


def test_steps_alias_and_lora_map():
    request, _, _ = build(
        {"prompt": "cat", "num_inference_steps": 12, "loras": {"kemono": 0.7}, "seed": 7}
    )
    assert request.steps == 12
    assert request.loras == {"kemono": 0.7}
    assert request.seed == 7


def test_conflicting_steps_rejected():
    try:
        ImageGenerationRequest(prompt="cat", steps=10, num_inference_steps=20)
    except Exception:
        return
    raise AssertionError("conflicting steps should be rejected")


def test_config_env_and_overrides():
    import os

    defaults = load_config(None, {"port": 9000})
    assert defaults.port == 9000
    assert defaults.quantize is True
    assert defaults.qtype == "convrot8"
    assert defaults.lora_mode == "hook"

    os.environ["MOSS_DEFAULT_STEPS"] = "24"
    try:
        assert load_config(None, None).default_steps == 24
    finally:
        del os.environ["MOSS_DEFAULT_STEPS"]

    try:
        Config(lora_mode="nope").validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid lora_mode should be rejected")


def test_config_coerces_lists_and_bools():
    config = load_config(
        None,
        {
            "default_uncond_skip_layers": "22,24",
            "loras": ["a.safetensors", "b.safetensors"],
            "lora_dirs": "X:/one;X:/two",
            "quantize": "false",
            "port": "8123",
            "layer_offloading_text_encoder_percent": "0.5",
        },
    )
    assert config.default_uncond_skip_layers == [22, 24]
    assert config.loras == ["a.safetensors", "b.safetensors"]
    assert config.lora_dirs == ["X:/one", "X:/two"]
    assert config.quantize is False
    assert config.port == 8123
    assert config.layer_offloading_text_encoder_percent == 0.5


def test_example_toml_parses_and_validates():
    example = Path(__file__).resolve().parents[1] / "config.example.toml"
    config = load_config(example, None).resolve(Path.cwd()).validate()
    assert config.aitk_root and Path(config.aitk_root, "toolkit").is_dir()
    assert Path(config.model_dir, "transformer").is_dir()
    assert config.validate() is config
    print(f"  model_dir={config.model_dir} qtype={config.qtype} loras={len(config.loras)}")


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
