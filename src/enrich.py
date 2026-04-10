"""Signal flagging and filtering for match clusters."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterable

from .match import MatchCluster
from .normalize import NormalizedRecord

log = logging.getLogger(__name__)

# Ordered by priority: earlier entries win when assigning a single signal_type.
SIGNAL_PRIORITY = [
    "new_approval",
    "pdufa_passed_no_action",
    "pdufa_imminent",
    "results_posted",
    "trial_complete",
    "trial_terminated",
    "trial_recruiting",
]


def _classify_record_signal(rec: NormalizedRecord, today: date) -> str | None:
    if rec.source == "openfda" and rec.approval_date:
        return "new_approval"
    if rec.source == "pdufa" and rec.pdufa_date:
        try:
            target = datetime.fromisoformat(rec.pdufa_date).date()
        except ValueError:
            target = None
        if target:
            if target < today:
                return "pdufa_passed_no_action"
            if target <= today + timedelta(days=14):
                return "pdufa_imminent"
    if rec.source == "clinicaltrials":
        status = (rec.status or "").upper()
        if rec.results_posted:
            return "results_posted"
        if status == "COMPLETED":
            return "trial_complete"
        if status in {"TERMINATED", "WITHDRAWN"}:
            return "trial_terminated"
        if status == "RECRUITING":
            return "trial_recruiting"
    return None


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d).date()
    except ValueError:
        return None


def apply_signals(clusters: list[MatchCluster], cfg: dict) -> list[MatchCluster]:
    today = date.today()
    for cluster in clusters:
        # Per-record signal tagging
        for rec in cluster.records:
            rec.signal_type = _classify_record_signal(rec, today)

        # Cluster-level flags
        phases_present = {r.phase for r in cluster.records if r.phase}
        statuses_present = {(r.status or "").upper() for r in cluster.records if r.status}

        has_active_phase3 = any(
            r.phase == "PHASE3" and (r.status or "").upper() in {"ACTIVE_NOT_RECRUITING", "RECRUITING", "COMPLETED"}
            for r in cluster.records
        )

        pdufa_dates = [_parse_iso(r.pdufa_date) for r in cluster.records if r.pdufa_date]
        pdufa_dates = [d for d in pdufa_dates if d is not None]
        has_approval = any(r.approval_date for r in cluster.records if r.source == "openfda")

        flags: dict = {}
        flags["high_signal"] = len(cluster.sources) >= 2

        if pdufa_dates:
            nearest = min(pdufa_dates)
            if today <= nearest <= today + timedelta(days=30) and has_active_phase3:
                flags["watch_list"] = True
            if nearest < today and not has_approval:
                flags["gap_alert"] = True
                flags["days_overdue"] = (today - nearest).days
                flags["expected_pdufa"] = nearest.isoformat()

        if phases_present and phases_present.issubset({"PHASE1", "PHASE2"}):
            flags["early_pipeline"] = True

        if statuses_present & {"TERMINATED", "WITHDRAWN"}:
            flags["negative_signal"] = True

        cluster.flags = flags

    n_high = sum(1 for c in clusters if c.flags.get("high_signal"))
    n_watch = sum(1 for c in clusters if c.flags.get("watch_list"))
    n_gap = sum(1 for c in clusters if c.flags.get("gap_alert"))
    log.info(
        "enriched %d clusters (high_signal=%d, watch_list=%d, gap_alert=%d)",
        len(clusters),
        n_high,
        n_watch,
        n_gap,
    )
    return clusters


def _cluster_matches_therapeutic_areas(cluster: MatchCluster, areas: Iterable[str]) -> bool:
    wanted = {a.lower() for a in areas if a}
    if not wanted:
        return True
    for r in cluster.records:
        if r.therapeutic_area and r.therapeutic_area.lower() in wanted:
            return True
    return False


def _cluster_matches_sponsors(cluster: MatchCluster, sponsors: Iterable[str]) -> bool:
    wanted = [s.lower() for s in sponsors if s]
    if not wanted:
        return True
    for r in cluster.records:
        sponsor = (r.sponsor or "").lower()
        if any(w in sponsor for w in wanted):
            return True
    canonical = (cluster.canonical_sponsor or "").lower()
    return any(w in canonical for w in wanted)


def apply_filters(clusters: list[MatchCluster], cfg: dict) -> list[MatchCluster]:
    areas = cfg.get("therapeutic_areas") or []
    sponsors = cfg.get("sponsors") or []
    if not areas and not sponsors:
        return clusters
    filtered = [
        c
        for c in clusters
        if _cluster_matches_therapeutic_areas(c, areas)
        and _cluster_matches_sponsors(c, sponsors)
    ]
    log.info("filtered %d -> %d clusters (areas=%s, sponsors=%s)", len(clusters), len(filtered), areas, sponsors)
    return filtered
