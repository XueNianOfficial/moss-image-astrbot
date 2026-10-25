"""A/B check: same prompt/seed with and without a LoRA (ai-toolkit engine).

    python scripts/quality_check.py --lora <file> --lora-scale 0.7 --size 512 --steps 12

Writes ``<outdir>/base.png`` and ``<outdir>/lora.png`` plus a PSNR comparison,
so a LoRA's effect (and the quantisation's stability) can be checked cheaply.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from moss_server.config import load_config  # noqa: E402
from moss_server.engine import GenerationRequest, MossEngine  # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MOSS_MODEL_DIR", ""), help="MOSS-Image diffusers bundle")
    parser.add_argument("--aitk-root", default=os.environ.get("AI_TOOLKIT_DIR", ""), help="ai-toolkit checkout")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--lora-scale", type=float, default=0.7)
    parser.add_argument("--prompt", default="a white wolf beastman with red eyes, anime style, detailed fur")
    parser.add_argument("--outdir", default="outputs/quality")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")

    overrides = {
        "model_dir": args.model_dir,
        "aitk_root": args.aitk_root,
        "warmup": False,
        "loras": [],
    }
    config = load_config(None, overrides).resolve(Path.cwd()).validate()

    engine = MossEngine(config)
    engine.ensure_loaded()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def render(tag: str):
        started = time.perf_counter()
        images = engine.generate(
            GenerationRequest(
                prompt=args.prompt, height=args.size, width=args.size, steps=args.steps,
                guidance_scale=3.0, seed=args.seed,
            )
        )
        elapsed = time.perf_counter() - started
        path = outdir / f"{tag}.png"
        images[0].image.save(path)
        print(f"{path}  {elapsed:.1f}s")
        return np.asarray(images[0].image.convert("RGB"))

    base = render("base")
    if args.lora:
        engine.loras.load(Path(args.lora).stem, args.lora, scale=args.lora_scale)
        lora = render("lora")
        print(f"psnr(base, lora) = {psnr(base, lora):.2f} dB   (low = LoRA has a strong effect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
