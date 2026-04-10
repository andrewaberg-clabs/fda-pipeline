"""Config loading and CLI override merge."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def merge_cli_overrides(cfg: dict[str, Any], cli_kwargs: dict[str, Any]) -> dict[str, Any]:
    """CLI overrides win when not None/empty tuple. Unknown CLI kwargs are ignored."""
    simple_keys = {
        "lookback_days": "lookback_days",
        "lookahead_days": "lookahead_days",
        "output_format": "output_format",
    }
    for cli_key, cfg_key in simple_keys.items():
        val = cli_kwargs.get(cli_key)
        if val is not None:
            cfg[cfg_key] = val

    # Multi-option CLI args arrive as tuples; empty tuple = no override.
    therapeutic_areas = cli_kwargs.get("therapeutic_areas")
    if therapeutic_areas:
        cfg["therapeutic_areas"] = list(therapeutic_areas)
    sponsors = cli_kwargs.get("sponsors")
    if sponsors:
        cfg["sponsors"] = list(sponsors)

    return cfg
