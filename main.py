"""CLI entrypoint for the FDA drug approval pipeline tracker."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src import diff, enrich, fetch, match, normalize, pdufa, report
from src.config import load_config, merge_cli_overrides
from src.logging_setup import setup_logging

log = logging.getLogger("main")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--lookback-days", type=int, default=None, help="Override config lookback_days.")
@click.option("--lookahead-days", type=int, default=None, help="Override config lookahead_days.")
@click.option(
    "--therapeutic-area",
    "therapeutic_areas",
    multiple=True,
    help="Filter clusters to these therapeutic areas (repeatable).",
)
@click.option(
    "--sponsor",
    "sponsors",
    multiple=True,
    help="Filter clusters by sponsor (case-insensitive substring, repeatable).",
)
@click.option(
    "--output-format",
    type=click.Choice(["md", "json", "both"]),
    default=None,
    help="Override config output_format.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default="config.yaml",
    show_default=True,
)
@click.option("--dry-run", is_flag=True, help="Replay most recent cached raw JSON dumps (no network).")
@click.option("--generate-content", is_flag=True, help="Generate CSV, newsletter, and LinkedIn content after the pipeline run.")
@click.option("--content-only", is_flag=True, help="Skip the pipeline and regenerate content from the most recent report.")
def cli(
    lookback_days,
    lookahead_days,
    therapeutic_areas,
    sponsors,
    output_format,
    config_path,
    dry_run,
    generate_content,
    content_only,
):
    """Run the full FDA pipeline tracker: fetch → normalize → match → enrich → report."""
    cfg = load_config(config_path)
    cfg = merge_cli_overrides(
        cfg,
        {
            "lookback_days": lookback_days,
            "lookahead_days": lookahead_days,
            "therapeutic_areas": therapeutic_areas,
            "sponsors": sponsors,
            "output_format": output_format,
        },
    )

    setup_logging(cfg["paths"]["logs_dir"])

    # Content-only mode: skip pipeline, regenerate content from most recent report.
    if content_only:
        from src.content import find_latest_report, generate_content as gen_content

        report_path = find_latest_report(cfg["paths"]["reports_dir"])
        if report_path is None:
            click.echo(
                f"No reports found in {cfg['paths']['reports_dir']} — run the pipeline first.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Regenerating content from: {report_path}")
        content_paths = gen_content(str(report_path), cfg)
        for channel, path in content_paths.items():
            click.echo(f"  {channel}: {path}")
        return

    log.info("starting pipeline: lookback=%s lookahead=%s dry_run=%s",
             cfg.get("lookback_days"), cfg.get("lookahead_days"), dry_run)

    try:
        report_path = run_pipeline(cfg, dry_run=dry_run)
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline failed: %s", e)
        sys.exit(1)

    click.echo(f"Report written: {report_path}")

    if generate_content:
        from src.content import generate_content as gen_content
        content_paths = gen_content(str(report_path), cfg)
        for channel, path in content_paths.items():
            click.echo(f"  {channel}: {path}")


def run_pipeline(cfg: dict, dry_run: bool = False) -> Path:
    # 1. Fetch
    raw = fetch.fetch_all(cfg, dry_run=dry_run)
    source_counts = {
        "clinicaltrials": len(raw.get("ct", [])),
        "openfda": len(raw.get("openfda", [])),
        "pdufa": len(raw.get("pdufa", [])),
    }

    # 2. Normalize
    records = normalize.normalize_all(raw, lookback_days=int(cfg.get("lookback_days", 90)))

    # 3. Match
    threshold = int(cfg.get("fuzzy_threshold", 85))
    clusters = match.cluster_records(records, threshold=threshold)

    # 4. Enrich (signal flags)
    clusters = enrich.apply_signals(clusters, cfg)

    # 5. PDUFA calendar sync (cross-reference with openFDA approvals)
    openfda_records = [r for r in records if r.source == "openfda"]
    calendar = pdufa.sync_calendar(raw.get("pdufa", []), cfg, approvals=openfda_records)

    # 6. Run-over-run diff
    prev_report = diff.load_previous_report(cfg["paths"]["reports_dir"])
    delta = diff.compute_diff(prev_report, clusters)

    # 7. Filter + Report
    filtered = enrich.apply_filters(clusters, cfg)
    return report.generate_report(
        filtered,
        calendar,
        cfg,
        warnings=raw.get("warnings", []),
        source_counts=source_counts,
        delta=delta,
    )


if __name__ == "__main__":
    cli()
