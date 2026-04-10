"""Logging setup: stdout + per-day file handler."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path


def setup_logging(logs_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    log_file = logs_path / f"fda_pipeline_{date.today().isoformat()}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers so repeated calls don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return root
