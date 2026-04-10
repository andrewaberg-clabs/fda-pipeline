"""Persistent PDUFA calendar management.

Stores ``data/pdufa_calendar.json`` keyed by stable entry_id, tracks status
transitions across runs, and cross-references openFDA approvals to mark
entries as resolved. Entries are never deleted; history appended instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from thefuzz import fuzz

from .normalize import NormalizedRecord, _parse_date

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
IMMINENT_WINDOW_DAYS = 14


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_entry_id(drug_name: str, nda_bla: Optional[str], pdufa_date: str) -> str:
    norm = (drug_name or "").lower().strip()
    key = f"{norm}|{nda_bla or ''}|{pdufa_date}".encode()
    return hashlib.sha1(key).hexdigest()[:12]


def _classify_status(pdufa_date: str, today: date, resolved: bool) -> str:
    if resolved:
        return "resolved"
    try:
        target = datetime.fromisoformat(pdufa_date).date()
    except ValueError:
        return "upcoming"
    if target < today:
        return "passed_no_action"
    if target <= today + timedelta(days=IMMINENT_WINDOW_DAYS):
        return "imminent"
    return "upcoming"


def _load_calendar(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "last_run": None, "entries": {}}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("pdufa calendar unreadable (%s); starting fresh", e)
        return {"schema_version": SCHEMA_VERSION, "last_run": None, "entries": {}}
    if "entries" not in data:
        data["entries"] = {}
    return data


def _save_calendar(path: Path, calendar: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(calendar, indent=2, default=str))
    os.replace(tmp, path)


def _append_history(entry: dict, from_status: Optional[str], to_status: str, note: str) -> None:
    entry.setdefault("history", []).append(
        {"ts": _utc_now(), "from": from_status, "to": to_status, "note": note}
    )


def _find_resolving_approval(
    entry: dict,
    approvals: list[NormalizedRecord],
    fuzzy_threshold: int = 85,
) -> Optional[NormalizedRecord]:
    """Return the first openFDA approval record that matches this entry."""
    entry_appnum = (entry.get("nda_bla_number") or "").upper().replace(" ", "")
    entry_name = (entry.get("drug_name") or "").lower()
    for rec in approvals:
        if rec.source != "openfda" or not rec.approval_date:
            continue
        rec_appnum = (rec.nda_bla_number or "").upper().replace(" ", "")
        if entry_appnum and rec_appnum and entry_appnum == rec_appnum:
            return rec
        if entry_name and rec.drug_name:
            score = fuzz.token_set_ratio(entry_name, rec.drug_name.lower())
            if score >= fuzzy_threshold:
                return rec
    return None


def sync_calendar(
    scraped_rows: list[dict],
    cfg: dict,
    approvals: Optional[list[NormalizedRecord]] = None,
) -> dict:
    """Load, merge, reclassify, cross-reference, and save the PDUFA calendar."""
    path = Path(cfg["paths"]["pdufa_calendar"])
    calendar = _load_calendar(path)
    entries: dict[str, dict] = calendar["entries"]
    today = date.today()
    now = _utc_now()

    # Pass 1: merge newly scraped rows
    for row in scraped_rows:
        drug_name = (row.get("drug_name") or "").strip()
        pdufa_date_iso = _parse_date(row.get("pdufa_date"))
        if not drug_name or not pdufa_date_iso:
            continue
        appnum = (row.get("nda_bla_number") or "").strip() or None
        if appnum:
            appnum = appnum.replace(" ", "").upper()
        entry_id = _make_entry_id(drug_name, appnum, pdufa_date_iso)

        if entry_id not in entries:
            # Check for a moved-date situation: same drug+appnum, different pdufa_date.
            for existing_id, existing in list(entries.items()):
                if (
                    existing.get("drug_name", "").lower() == drug_name.lower()
                    and (existing.get("nda_bla_number") or None) == appnum
                    and existing.get("pdufa_date") != pdufa_date_iso
                    and existing.get("status") not in {"resolved", "superseded"}
                ):
                    prev_status = existing.get("status")
                    existing["status"] = "superseded"
                    existing["last_checked"] = now
                    existing["superseded_by"] = entry_id
                    _append_history(
                        existing,
                        prev_status,
                        "superseded",
                        f"replaced by new PDUFA date {pdufa_date_iso}",
                    )

            new_status = _classify_status(pdufa_date_iso, today, resolved=False)
            entry = {
                "entry_id": entry_id,
                "drug_name": drug_name,
                "applicant": (row.get("applicant") or "").strip() or None,
                "nda_bla_number": appnum,
                "submission_type": row.get("submission_type"),
                "pdufa_date": pdufa_date_iso,
                "indication": (row.get("indication") or "").strip() or None,
                "submission_date": _parse_date(row.get("submission_date")),
                "status": new_status,
                "first_seen": now,
                "last_checked": now,
                "last_source": "fda_pdufa_page",
                "resolved_approval_date": None,
                "history": [],
            }
            _append_history(entry, None, new_status, "first_seen")
            entries[entry_id] = entry
        else:
            entry = entries[entry_id]
            prev_status = entry.get("status")
            entry["last_checked"] = now
            entry["last_source"] = "fda_pdufa_page"
            # Refresh mutable descriptive fields if the scrape has better info.
            for k in ("applicant", "submission_type", "indication", "submission_date"):
                if row.get(k) and not entry.get(k):
                    entry[k] = row[k] if k != "submission_date" else _parse_date(row[k])
            new_status = _classify_status(
                entry["pdufa_date"], today, resolved=prev_status == "resolved"
            )
            if new_status != prev_status:
                entry["status"] = new_status
                _append_history(entry, prev_status, new_status, "status transition")
            else:
                _append_history(entry, prev_status, new_status, "refreshed")

    # Pass 2: recompute status for every existing entry (handles time passage)
    for entry in entries.values():
        if entry.get("status") in {"resolved", "superseded"}:
            continue
        prev = entry.get("status")
        new_status = _classify_status(entry["pdufa_date"], today, resolved=False)
        if new_status != prev:
            entry["status"] = new_status
            _append_history(entry, prev, new_status, "time-based reclassification")

    # Pass 3: cross-reference openFDA approvals
    if approvals:
        threshold = int(cfg.get("fuzzy_threshold", 85))
        for entry in entries.values():
            if entry.get("status") in {"resolved", "superseded"}:
                continue
            match = _find_resolving_approval(entry, approvals, fuzzy_threshold=threshold)
            if match:
                prev = entry.get("status")
                entry["status"] = "resolved"
                entry["resolved_approval_date"] = match.approval_date
                entry["last_checked"] = now
                _append_history(
                    entry,
                    prev,
                    "resolved",
                    f"matched openFDA approval {match.nda_bla_number or match.drug_name}",
                )

    calendar["schema_version"] = SCHEMA_VERSION
    calendar["last_run"] = now
    _save_calendar(path, calendar)

    status_counts: dict[str, int] = {}
    for e in entries.values():
        status_counts[e.get("status", "unknown")] = status_counts.get(e.get("status", "unknown"), 0) + 1
    log.info("pdufa calendar: %d entries (%s)", len(entries), status_counts)

    return calendar
