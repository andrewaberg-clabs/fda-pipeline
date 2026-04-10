"""Markdown + JSON report generation."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from .match import MatchCluster
from .normalize import NormalizedRecord

log = logging.getLogger(__name__)

STATUS_BADGES = {
    "upcoming": "🟢 upcoming",
    "imminent": "🟡 imminent",
    "passed_no_action": "🔴 passed_no_action",
    "resolved": "✅ resolved",
    "superseded": "⚪ superseded",
}


def _badge(status: Optional[str]) -> str:
    return STATUS_BADGES.get(status or "", f"⚪ {status or 'unknown'}")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No records._\n"
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        cells = [str(c).replace("|", "\\|") if c is not None else "" for c in r]
        out += "| " + " | ".join(cells) + " |\n"
    return out + "\n"


def _records_by_signal(clusters: list[MatchCluster], signal: str) -> list[NormalizedRecord]:
    out: list[NormalizedRecord] = []
    for c in clusters:
        for r in c.records:
            if r.signal_type == signal:
                out.append(r)
    return out


def _records_by_phase(clusters: list[MatchCluster], phase: str) -> list[tuple[MatchCluster, NormalizedRecord]]:
    out = []
    for c in clusters:
        for r in c.records:
            if r.source == "clinicaltrials" and r.phase == phase:
                out.append((c, r))
    return out


def _terminated_records(clusters: list[MatchCluster]) -> list[tuple[MatchCluster, NormalizedRecord]]:
    out = []
    for c in clusters:
        for r in c.records:
            if r.source == "clinicaltrials" and (r.status or "").upper() in {"TERMINATED", "WITHDRAWN"}:
                out.append((c, r))
    return out


def _calendar_entries_in_window(calendar: dict, days: int) -> list[dict]:
    entries = list(calendar.get("entries", {}).values())
    today = date.today()
    cutoff = today + timedelta(days=days)
    in_window = []
    for e in entries:
        try:
            d = datetime.fromisoformat(e["pdufa_date"]).date()
        except (ValueError, KeyError, TypeError):
            continue
        if d <= cutoff and e.get("status") not in {"superseded"}:
            in_window.append(e)
    in_window.sort(key=lambda x: x.get("pdufa_date", ""))
    return in_window


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_exec_summary(clusters: list[MatchCluster], calendar: dict) -> str:
    today = date.today()
    entries = list(calendar.get("entries", {}).values())
    counts = {"upcoming": 0, "imminent": 0, "passed_no_action": 0, "resolved": 0}
    for e in entries:
        s = e.get("status")
        if s in counts:
            counts[s] += 1

    new_approvals = len(_records_by_signal(clusters, "new_approval"))
    p1 = sum(1 for c in clusters for r in c.records if r.source == "clinicaltrials" and r.phase == "PHASE1")
    p2 = sum(1 for c in clusters for r in c.records if r.source == "clinicaltrials" and r.phase == "PHASE2")
    p3 = sum(1 for c in clusters for r in c.records if r.source == "clinicaltrials" and r.phase == "PHASE3")
    terminated = len(_terminated_records(clusters))
    high_signal = sum(1 for c in clusters if c.flags.get("high_signal"))
    gap_alerts = sum(1 for c in clusters if c.flags.get("gap_alert"))

    md = "## 1. Executive Summary\n\n"
    md += f"- **{new_approvals}** new FDA approvals (NDA/BLA) in the lookback window\n"
    md += f"- **{p1}** Phase 1, **{p2}** Phase 2, **{p3}** Phase 3 trial records\n"
    md += f"- **{terminated}** terminated/withdrawn trial records\n"
    md += (
        f"- **{sum(counts.values())}** total PDUFA calendar entries "
        f"({counts['upcoming']} upcoming, {counts['imminent']} imminent, "
        f"{counts['passed_no_action']} passed_no_action, {counts['resolved']} resolved)\n"
    )
    md += f"- **{high_signal}** high-signal clusters (present in 2+ sources)\n"
    md += f"- **{gap_alerts}** gap alerts (PDUFA passed with no visible FDA action)\n\n"
    return md


def _render_pdufa_calendar(calendar: dict, days: int, title_suffix: str) -> str:
    rows = []
    for e in _calendar_entries_in_window(calendar, days):
        rows.append(
            [
                _badge(e.get("status")),
                e.get("drug_name", ""),
                e.get("applicant") or "",
                e.get("submission_type") or "",
                e.get("pdufa_date", ""),
                (e.get("indication") or "")[:80],
                e.get("nda_bla_number") or "",
            ]
        )
    md = f"## 3. PDUFA Calendar — {title_suffix}\n\n"
    md += _md_table(
        ["Status", "Drug", "Applicant", "Type", "PDUFA Date", "Indication", "NDA/BLA"],
        rows,
    )
    return md


def _render_watchlist(clusters: list[MatchCluster]) -> str:
    rows = []
    for c in clusters:
        if not c.flags.get("watch_list"):
            continue
        pdufa_date = next((r.pdufa_date for r in c.records if r.pdufa_date), "")
        nct = next((r.nct_id for r in c.records if r.nct_id), "") or ""
        rows.append(
            [
                c.canonical_name,
                c.canonical_sponsor or "",
                pdufa_date,
                nct,
                ", ".join(sorted(c.sources)),
            ]
        )
    md = "## 4. PDUFA Watch List (≤30 days + active Phase 3)\n\n"
    md += _md_table(["Drug", "Sponsor", "PDUFA", "Phase 3 NCT", "Sources"], rows)
    return md


def _render_gap_alerts(clusters: list[MatchCluster]) -> str:
    rows = []
    for c in clusters:
        if not c.flags.get("gap_alert"):
            continue
        rows.append(
            [
                c.canonical_name,
                c.canonical_sponsor or "",
                c.flags.get("expected_pdufa", ""),
                str(c.flags.get("days_overdue", "")),
                c.nda_bla_number or "",
            ]
        )
    md = "## 5. Gap Alerts (PDUFA passed, no approval observed)\n\n"
    md += _md_table(["Drug", "Sponsor", "Expected PDUFA", "Days Overdue", "NDA/BLA"], rows)
    return md


def _render_new_approvals(clusters: list[MatchCluster]) -> str:
    rows = []
    seen: set[str] = set()
    for c in clusters:
        for r in c.records:
            if r.signal_type != "new_approval":
                continue
            key = f"{r.nda_bla_number}|{r.approval_date}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                [
                    r.approval_date or "",
                    r.drug_name or "",
                    r.generic_name or "",
                    r.sponsor or "",
                    r.nda_bla_number or "",
                    (r.indication or r.therapeutic_area or "")[:60],
                ]
            )
    rows.sort(key=lambda x: x[0], reverse=True)
    md = "## 6. New FDA Approvals\n\n"
    md += _md_table(["Approval Date", "Brand", "Generic", "Sponsor", "NDA/BLA", "Area"], rows)
    return md


def _render_phase_table(clusters: list[MatchCluster], phase: str, section_num: int, title: str) -> str:
    rows = []
    seen: set[str] = set()
    for c, r in _records_by_phase(clusters, phase):
        if r.nct_id and r.nct_id in seen:
            continue
        seen.add(r.nct_id or "")
        rows.append(
            [
                r.nct_id or "",
                r.drug_name or c.canonical_name,
                r.sponsor or "",
                r.status or "",
                r.completion_date or "",
                "✓" if r.results_posted else "",
            ]
        )
    md = f"## {section_num}. {title}\n\n"
    md += _md_table(["NCT", "Drug", "Sponsor", "Status", "Completion", "Results"], rows)
    return md


def _render_terminated(clusters: list[MatchCluster]) -> str:
    rows = []
    seen: set[str] = set()
    for c, r in _terminated_records(clusters):
        if r.nct_id and r.nct_id in seen:
            continue
        seen.add(r.nct_id or "")
        rows.append(
            [
                r.nct_id or "",
                r.drug_name or c.canonical_name,
                r.sponsor or "",
                r.phase or "",
                r.status or "",
            ]
        )
    md = "## 10. Terminated / Withdrawn Programs\n\n"
    md += _md_table(["NCT", "Drug", "Sponsor", "Phase", "Status"], rows)
    return md


def _render_high_signal(clusters: list[MatchCluster]) -> str:
    rows = []
    for c in sorted(clusters, key=lambda x: (-len(x.sources), -x.match_confidence)):
        if not c.flags.get("high_signal"):
            continue
        rows.append(
            [
                c.canonical_name,
                "+".join(sorted(c.sources)),
                c.canonical_sponsor or "",
                f"{c.match_confidence:.1f}",
            ]
        )
    md = "## 11. High-Signal Matches\n\n"
    md += _md_table(["Drug", "Sources", "Sponsor", "Confidence"], rows)
    return md


def _render_appendix(
    calendar: dict,
    cfg: dict,
    warnings: list[str],
    source_counts: dict,
) -> str:
    md = "## 12. Appendix\n\n"
    md += "### Source Fetch Counts\n\n"
    for src, count in source_counts.items():
        md += f"- **{src}**: {count} records\n"
    md += "\n### Warnings\n\n"
    if warnings:
        for w in warnings:
            md += f"- {w}\n"
    else:
        md += "_No warnings._\n"

    md += "\n### Full PDUFA Calendar (next "
    lookahead = int(cfg.get("lookahead_days", 365))
    md += f"{lookahead} days)\n\n"
    entries = _calendar_entries_in_window(calendar, lookahead)
    rows = []
    for e in entries:
        rows.append(
            [
                _badge(e.get("status")),
                e.get("drug_name", ""),
                e.get("applicant") or "",
                e.get("submission_type") or "",
                e.get("pdufa_date", ""),
                (e.get("indication") or "")[:60],
                e.get("nda_bla_number") or "",
            ]
        )
    md += _md_table(
        ["Status", "Drug", "Applicant", "Type", "PDUFA Date", "Indication", "NDA/BLA"], rows
    )

    md += "\n### Config Snapshot\n\n```yaml\n"
    safe_cfg = {
        "lookback_days": cfg.get("lookback_days"),
        "lookahead_days": cfg.get("lookahead_days"),
        "fuzzy_threshold": cfg.get("fuzzy_threshold"),
        "therapeutic_areas": cfg.get("therapeutic_areas"),
        "sponsors": cfg.get("sponsors"),
        "output_format": cfg.get("output_format"),
    }
    for k, v in safe_cfg.items():
        md += f"{k}: {v}\n"
    md += "```\n"
    return md


# ---------------------------------------------------------------------------
# Top-level writers
# ---------------------------------------------------------------------------


def _write_markdown(
    clusters: list[MatchCluster],
    calendar: dict,
    cfg: dict,
    warnings: list[str],
    run_date: str,
    source_counts: dict,
) -> Path:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"fda_pipeline_{run_date}.md"

    lookback = cfg.get("lookback_days", "?")
    lookahead = cfg.get("lookahead_days", "?")
    filters: list[str] = []
    if cfg.get("therapeutic_areas"):
        filters.append(f"therapeutic_area={','.join(cfg['therapeutic_areas'])}")
    if cfg.get("sponsors"):
        filters.append(f"sponsor={','.join(cfg['sponsors'])}")

    md = "# FDA Pipeline Intelligence Report\n\n"
    md += f"**Run date:** {run_date}\n"
    md += f"**Lookback:** {lookback} days | **Lookahead:** {lookahead} days\n"
    md += f"**Filters:** {', '.join(filters) if filters else 'none'}\n\n"

    stale = any("stale" in w.lower() or "scrape" in w.lower() for w in warnings)
    if stale:
        md += "> ⚠️ **PDUFA data may be stale** — one or more scrape sources returned 0 rows. "
        md += "Calendar reflects the most recent successful run.\n\n"

    md += _render_exec_summary(clusters, calendar)
    md += "## 2. Pipeline Overview\n\n_See sections below for per-phase tables._\n\n"
    md += _render_pdufa_calendar(calendar, 90, "Next 90 Days")
    md += _render_watchlist(clusters)
    md += _render_gap_alerts(clusters)
    md += _render_new_approvals(clusters)
    md += _render_phase_table(clusters, "PHASE3", 7, "Late-Stage Trial Activity (Phase 3)")
    md += _render_phase_table(clusters, "PHASE2", 8, "Mid-Stage Pipeline (Phase 2)")
    md += _render_phase_table(clusters, "PHASE1", 9, "Early Pipeline (Phase 1)")
    md += _render_terminated(clusters)
    md += _render_high_signal(clusters)
    md += _render_appendix(calendar, cfg, warnings, source_counts)

    out.write_text(md)
    log.info("wrote markdown report %s", out)
    return out


def _write_json(
    clusters: list[MatchCluster],
    calendar: dict,
    cfg: dict,
    warnings: list[str],
    run_date: str,
    source_counts: dict,
) -> Path:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"fda_pipeline_{run_date}.json"

    payload = {
        "run_date": run_date,
        "config": {
            "lookback_days": cfg.get("lookback_days"),
            "lookahead_days": cfg.get("lookahead_days"),
            "fuzzy_threshold": cfg.get("fuzzy_threshold"),
            "therapeutic_areas": cfg.get("therapeutic_areas"),
            "sponsors": cfg.get("sponsors"),
            "output_format": cfg.get("output_format"),
        },
        "source_counts": source_counts,
        "warnings": warnings,
        "clusters": [c.to_dict() for c in clusters],
        "pdufa_calendar": calendar,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote json report %s", out)
    return out


def generate_report(
    clusters: list[MatchCluster],
    calendar: dict,
    cfg: dict,
    warnings: list[str],
    source_counts: dict | None = None,
) -> Path:
    run_date = date.today().isoformat()
    output_format = cfg.get("output_format", "both")
    source_counts = source_counts or {}

    last: Path = Path(cfg["paths"]["reports_dir"])
    if output_format in {"md", "both"}:
        last = _write_markdown(clusters, calendar, cfg, warnings, run_date, source_counts)
    if output_format in {"json", "both"}:
        last = _write_json(clusters, calendar, cfg, warnings, run_date, source_counts)
    return last
