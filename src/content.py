"""Content generation: CSV export, newsletter, and LinkedIn posts from pipeline data.

Reads a JSON report produced by ``report.generate_report`` and generates
channel-specific content files to the ``content/`` directory.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "cluster_id",
    "canonical_name",
    "canonical_sponsor",
    "sources",
    "high_signal",
    "match_confidence",
    "source",
    "source_id",
    "drug_name",
    "generic_name",
    "sponsor",
    "phase",
    "status",
    "approval_date",
    "nct_id",
    "nda_bla_number",
    "therapeutic_area",
    "indication",
    "signal_type",
    "completion_date",
]


def _generate_csv(report: dict, output_dir: Path) -> Path:
    """Flatten clusters into one row per record for spreadsheet/BI import."""
    run_date = report.get("run_date", date.today().isoformat())
    out = output_dir / f"fda_data_{run_date}.csv"

    rows: list[dict] = []
    for cluster in report.get("clusters", []):
        cluster_base = {
            "cluster_id": cluster.get("cluster_id", ""),
            "canonical_name": cluster.get("canonical_name", ""),
            "canonical_sponsor": cluster.get("canonical_sponsor", ""),
            "sources": ",".join(cluster.get("sources", [])),
            "high_signal": cluster.get("flags", {}).get("high_signal", False),
            "match_confidence": cluster.get("match_confidence", ""),
        }
        for rec in cluster.get("records", []):
            row = {**cluster_base}
            for col in _CSV_COLUMNS:
                if col not in row:
                    row[col] = rec.get(col, "")
            rows.append(row)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("wrote CSV: %s (%d rows)", out, len(rows))
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_content(report_path: str, cfg: dict) -> dict[str, Path]:
    """Read a JSON report and generate content for all enabled channels.

    Returns a dict mapping channel name → output file path.
    """
    report = json.loads(Path(report_path).read_text())
    content_cfg = cfg.get("content", {})
    output_dir = Path(content_cfg.get("output_dir", "content"))
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # CSV (always available — no LLM needed)
    if content_cfg.get("csv", {}).get("enabled", True):
        paths["csv"] = _generate_csv(report, output_dir)

    # Newsletter (requires LLM — added in Commit 2)
    if content_cfg.get("newsletter", {}).get("enabled", True):
        try:
            from .llm import generate_newsletter_narrative
            paths["newsletter"] = _generate_newsletter(report, output_dir, cfg)
        except ImportError:
            log.info("newsletter generation skipped — llm module not yet available")
        except Exception as e:  # noqa: BLE001
            log.warning("newsletter generation failed: %s", e)

    # LinkedIn (requires LLM — added in Commit 3)
    if content_cfg.get("linkedin", {}).get("enabled", True):
        try:
            from .llm import generate_linkedin_posts
            paths["linkedin"] = _generate_linkedin(report, output_dir, cfg)
        except ImportError:
            log.info("linkedin generation skipped — llm module not yet available")
        except Exception as e:  # noqa: BLE001
            log.warning("linkedin generation failed: %s", e)

    return paths


def _generate_newsletter(report: dict, output_dir: Path, cfg: dict) -> Path:
    """Generate HTML newsletter. Implemented in Commit 2."""
    raise NotImplementedError("newsletter generation not yet implemented")


def _generate_linkedin(report: dict, output_dir: Path, cfg: dict) -> Path:
    """Generate LinkedIn posts. Implemented in Commit 3."""
    raise NotImplementedError("linkedin generation not yet implemented")
