"""Entry point: ``python -m moss_server [options]``."""
from __future__ import annotations

import logging
import sys

from .api import create_app
from .config import config_from_args

BANNER = r"""
  __  __  ___  ___ ___    ___ __  __
 |  \/  |/ _ \/ __/ __|  |_ _|  \/  |  MOSS-Image server (ai-toolkit engine)
 | |\/| | (_) \__ \__ \   | || |\/| |
 |_|  |_|\___/|___/___/  |___|_|  |_|
"""


def main(argv: list[str] | None = None) -> int:
    config, _ = config_from_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    print(BANNER)
    logging.getLogger(__name__).info(
        "model=%s qtype=%s device=%s listening on http://%s:%d",
        config.model_dir, config.qtype if config.quantize else "bf16", config.device, config.host, config.port,
    )

    import uvicorn  # noqa: PLC0415 - imported after logging is configured

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
