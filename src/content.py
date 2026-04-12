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

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


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

    # Newsletter (requires LLM + jinja2)
    if content_cfg.get("newsletter", {}).get("enabled", True):
        try:
            paths["newsletter"] = _generate_newsletter(report, output_dir, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("newsletter generation failed: %s", e)

    # LinkedIn (requires LLM)
    if content_cfg.get("linkedin", {}).get("enabled", True):
        try:
            paths["linkedin"] = _generate_linkedin(report, output_dir, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("linkedin generation failed: %s", e)

    return paths


def _generate_newsletter(report: dict, output_dir: Path, cfg: dict) -> Path:
    """Generate HTML newsletter using LLM narrative + Jinja2 template."""
    from .llm import generate_newsletter_narrative

    run_date = report.get("run_date", date.today().isoformat())

    # Get LLM narrative (or fallback to empty strings)
    narrative = generate_newsletter_narrative(report, cfg)
    if not narrative:
        narrative = {
            "opening": "This week's FDA pipeline data is available below.",
            "approvals": "See the approvals table for details.",
            "pipeline": "See the full report for trial status details.",
            "watch": "See the high-signal matches table below.",
        }

    # Extract data for template tables
    approvals = []
    high_signal = []
    for c in report.get("clusters", []):
        flags = c.get("flags", {})
        if flags.get("high_signal"):
            high_signal.append(c)
        for rec in c.get("records", []):
            if rec.get("signal_type") == "new_approval":
                approvals.append(rec)

    # Render template
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        log.warning("jinja2 not installed — run: pip install jinja2")
        raise

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("newsletter.html.j2")
    html = template.render(
        run_date=run_date,
        narrative=narrative,
        approvals=approvals[:20],
        high_signal=high_signal[:15],
        source_counts=report.get("source_counts", {}),
        total_clusters=len(report.get("clusters", [])),
    )

    out = output_dir / f"newsletter_{run_date}.html"
    out.write_text(html, encoding="utf-8")
    log.info("wrote newsletter: %s", out)
    return out


def _generate_linkedin(report: dict, output_dir: Path, cfg: dict) -> Path:
    """Generate LinkedIn post drafts as JSON."""
    from .llm import generate_linkedin_posts as llm_linkedin

    run_date = report.get("run_date", date.today().isoformat())
    posts = llm_linkedin(report, cfg)

    if posts is None:
        posts = []
        log.info("linkedin: no posts generated (LLM unavailable)")

    out = output_dir / f"linkedin_{run_date}.json"
    out.write_text(json.dumps(posts, indent=2), encoding="utf-8")
    log.info("wrote linkedin posts: %s (%d posts)", out, len(posts))
    return out


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------


def find_latest_report(reports_dir: str | Path) -> Path | None:
    """Return the newest ``fda_pipeline_*.json`` report, or None if none exist."""
    rdir = Path(reports_dir)
    candidates = sorted(rdir.glob("fda_pipeline_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _main(argv: list[str] | None = None) -> int:
    """Regenerate content from an existing report without running the pipeline.

    Usage:
        python -m src.content                                  # most recent report
        python -m src.content reports/fda_pipeline_YYYY-MM-DD.json
    """
    import argparse

    from .config import load_config
    from .logging_setup import setup_logging

    parser = argparse.ArgumentParser(
        prog="python -m src.content",
        description="Regenerate CSV, newsletter, and LinkedIn content from a report.",
    )
    parser.add_argument("report", nargs="?", help="Path to JSON report (default: most recent).")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg["paths"]["logs_dir"])

    if args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            log.error("report not found: %s", report_path)
            return 1
    else:
        report_path = find_latest_report(cfg["paths"]["reports_dir"])
        if report_path is None:
            log.error("no reports found in %s — run the pipeline first", cfg["paths"]["reports_dir"])
            return 1
        log.info("using most recent report: %s", report_path)

    paths = generate_content(str(report_path), cfg)
    for channel, path in paths.items():
        print(f"  {channel}: {path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
