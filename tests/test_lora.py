"""LoRA registry tests (ai-toolkit hook mode: nothing is merged)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from moss_server.lora import LoRARegistry, inspect_lora, scan_lora_dirs

REAL_LORA = Path(os.environ["MOSS_TEST_LORA"]) if os.environ.get("MOSS_TEST_LORA") else None


def write_lora(path: Path, rank: int = 4) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "diffusion_model.layers.0.proj.q.lora_A.weight": torch.randn(rank, 32),
            "diffusion_model.layers.0.proj.q.lora_B.weight": torch.randn(16, rank),
        },
        str(path),
    )


def test_register_and_specs():
    registry = LoRARegistry()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.safetensors"
        write_lora(path)
        entry = registry.load("demo", path, scale=0.7)
        assert entry.keys == 2, entry
        specs = registry.specs()
        assert specs == [{"path": str(path.resolve()), "strength": 0.7}], specs

        # per-request strengths override without touching the registry
        assert registry.specs({"demo": 0.2}) == [{"path": str(path.resolve()), "strength": 0.2}]
        assert registry.get("demo").scale == 0.7

        registry.set_scale("demo", 1.0)
        assert registry.specs()[0]["strength"] == 1.0

        assert registry.unload("demo") is True
        assert registry.specs() == []
        assert not registry.unload("demo")


def test_rejects_duplicates_and_missing_files():
    registry = LoRARegistry()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.safetensors"
        write_lora(path)
        registry.load("demo", path)
        try:
            registry.load("demo", path)
        except ValueError as error:
            assert "already" in str(error)
        else:
            raise AssertionError("duplicate name should be rejected")
        registry.load("demo", path, replace=True)

        try:
            registry.load("nope", Path(directory) / "missing.safetensors")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing file should be rejected")


def test_rejects_non_lora_safetensors():
    from safetensors.torch import save_file

    registry = LoRARegistry()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "not_a_lora.safetensors"
        save_file({"layers.0.proj.q.weight": torch.randn(4, 4)}, str(path))
        try:
            registry.load("bad", path)
        except ValueError as error:
            assert "no LoRA tensors" in str(error)
        else:
            raise AssertionError("non-LoRA file should be rejected")


def test_inspect_real_ai_toolkit_lora():
    if REAL_LORA is None or not REAL_LORA.is_file():
        print("  skipped: real LoRA not present")
        return
    keys = inspect_lora(REAL_LORA)
    assert keys >= 400, keys
    print(f"  {REAL_LORA.name}: {keys} LoRA tensors")


def test_scan_lora_dirs():
    scan_dir = os.environ.get("MOSS_TEST_LORA_DIR") or os.environ.get("AI_TOOLKIT_OUTPUT_DIR") or tempfile.gettempdir()
    found = scan_lora_dirs([scan_dir])
    if not found:
        print("  skipped: no LoRA files found")
        return
    assert {"name", "path", "size_mb", "modified"} <= set(found[0])
    print(f"  {len(found)} LoRA files, newest {found[0]['name']}")


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
