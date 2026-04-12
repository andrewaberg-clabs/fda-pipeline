"""Run-over-run diffing: compare current clusters to the most recent prior report.

Produces a compact delta — new trials, status changes, new approvals, and
entries that dropped out of the window — so the weekly report leads with
"what changed" instead of a full data dump.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .match import MatchCluster

log = logging.getLogger(__name__)


@dataclass
class DiffEntry:
    source_id: str
    drug_name: str | None = None
    sponsor: str | None = None
    phase: str | None = None
    nct_id: str | None = None
    nda_bla_number: str | None = None
    approval_date: str | None = None
    old_status: str | None = None
    new_status: str | None = None


@dataclass
class DiffReport:
    prev_run_date: str | None = None
    new_trials: list[DiffEntry] = field(default_factory=list)
    status_changes: list[DiffEntry] = field(default_factory=list)
    new_approvals: list[DiffEntry] = field(default_factory=list)
    dropped: list[DiffEntry] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_trials or self.status_changes or self.new_approvals or self.dropped)

    @property
    def is_first_run(self) -> bool:
        return self.prev_run_date is None


def load_previous_report(reports_dir: str) -> dict | None:
    """Find the most recent JSON report that predates today."""
    rdir = Path(reports_dir)
    today_str = date.today().isoformat()
    candidates = sorted(rdir.glob("fda_pipeline_*.json"), reverse=True)
    for path in candidates:
        # Extract date from filename: fda_pipeline_YYYY-MM-DD.json
        stem = path.stem  # "fda_pipeline_2026-04-10"
        file_date = stem.replace("fda_pipeline_", "")
        if file_date < today_str:
            try:
                data = json.loads(path.read_text())
                log.info("diff baseline: loaded %s", path.name)
                return data
            except (json.JSONDecodeError, OSError) as e:
                log.warning("failed to load previous report %s: %s", path, e)
    log.info("diff baseline: no previous report found (first run)")
    return None


def _extract_records(report: dict) -> dict[str, dict]:
    """Build {source_id: record_dict} from a serialized report."""
    records: dict[str, dict] = {}
    for cluster in report.get("clusters", []):
        for rec in cluster.get("records", []):
            sid = rec.get("source_id")
            if sid:
                records[sid] = rec
    return records


def _current_records(clusters: list[MatchCluster]) -> dict[str, dict]:
    """Build {source_id: record_fields} from live MatchClusters."""
    records: dict[str, dict] = {}
    for cluster in clusters:
        for rec in cluster.records:
            records[rec.source_id] = {
                "source_id": rec.source_id,
                "source": rec.source,
                "drug_name": rec.drug_name,
                "generic_name": rec.generic_name,
                "sponsor": rec.sponsor,
                "phase": rec.phase,
                "status": rec.status,
                "nct_id": rec.nct_id,
                "nda_bla_number": rec.nda_bla_number,
                "approval_date": rec.approval_date,
            }
    return records


def compute_diff(prev_report: dict | None, clusters: list[MatchCluster]) -> DiffReport:
    """Compare current clusters against the previous report's clusters."""
    if prev_report is None:
        return DiffReport()

    prev = _extract_records(prev_report)
    curr = _current_records(clusters)

    prev_ids = set(prev)
    curr_ids = set(curr)

    diff = DiffReport(prev_run_date=prev_report.get("run_date"))

    # New entries (in current, not in previous)
    for sid in sorted(curr_ids - prev_ids):
        r = curr[sid]
        entry = DiffEntry(
            source_id=sid,
            drug_name=r.get("drug_name"),
            sponsor=r.get("sponsor"),
            phase=r.get("phase"),
            nct_id=r.get("nct_id"),
            nda_bla_number=r.get("nda_bla_number"),
            approval_date=r.get("approval_date"),
            new_status=r.get("status"),
        )
        if r.get("source") == "openfda":
            diff.new_approvals.append(entry)
        else:
            diff.new_trials.append(entry)

    # Status changes (same source_id, different status)
    for sid in sorted(curr_ids & prev_ids):
        old_status = prev[sid].get("status")
        new_status = curr[sid].get("status")
        if old_status and new_status and old_status != new_status:
            r = curr[sid]
            diff.status_changes.append(DiffEntry(
                source_id=sid,
                drug_name=r.get("drug_name"),
                sponsor=r.get("sponsor"),
                phase=r.get("phase"),
                nct_id=r.get("nct_id"),
                old_status=old_status,
                new_status=new_status,
            ))

    # Dropped entries (in previous, not in current)
    for sid in sorted(prev_ids - curr_ids):
        r = prev[sid]
        diff.dropped.append(DiffEntry(
            source_id=sid,
            drug_name=r.get("drug_name"),
            sponsor=r.get("sponsor"),
            phase=r.get("phase"),
            nct_id=r.get("nct_id"),
            old_status=r.get("status"),
        ))

    log.info(
        "diff vs %s: +%d new trials, %d status changes, +%d approvals, -%d dropped",
        diff.prev_run_date,
        len(diff.new_trials),
        len(diff.status_changes),
        len(diff.new_approvals),
        len(diff.dropped),
    )
    return diff
