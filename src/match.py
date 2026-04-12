"""3-way fuzzy matching: cluster NormalizedRecords across sources.

Strategy:
    Stage A: bucket by NDA/BLA application number (exact match).
    Stage B: fuzzy-match remaining records by canonical drug name using
             thefuzz.token_set_ratio. Tiebreaker for borderline scores uses
             sponsor + indication fuzzy match.
    Stage C: every un-matched record becomes a singleton cluster so nothing
             is dropped from downstream reporting.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from thefuzz import fuzz

from .normalize import NormalizedRecord

log = logging.getLogger(__name__)


@dataclass
class MatchCluster:
    cluster_id: str
    canonical_name: str
    canonical_sponsor: Optional[str] = None
    nda_bla_number: Optional[str] = None
    sources: set[str] = field(default_factory=set)
    records: list[NormalizedRecord] = field(default_factory=list)
    match_confidence: float = 100.0
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "canonical_name": self.canonical_name,
            "canonical_sponsor": self.canonical_sponsor,
            "nda_bla_number": self.nda_bla_number,
            "sources": sorted(self.sources),
            "records": [r.to_dict() for r in self.records],
            "match_confidence": round(self.match_confidence, 2),
            "flags": self.flags,
        }


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

_SALT_SUFFIXES = [
    " hcl",
    " hydrochloride",
    " sodium",
    " potassium",
    " calcium",
    " acetate",
    " sulfate",
    " sulphate",
    " maleate",
    " citrate",
    " phosphate",
    " mesylate",
    " tartrate",
    " fumarate",
    " succinate",
]

_DOSAGE_SUFFIXES = [
    " for injection",
    " for oral suspension",
    " for inhalation",
    " injection",
    " tablet",
    " tablets",
    " capsule",
    " capsules",
    " solution",
    " suspension",
    " cream",
    " ointment",
    " gel",
    " patch",
    " powder",
    " spray",
    " inhaler",
    " ophthalmic",
    " subcutaneous",
    " intravenous",
]


def _canonicalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)  # drop parenthetical brand/generic pairs
    s = re.sub(r"\s+", " ", s).strip()
    for suffix in _DOSAGE_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    for salt in _SALT_SUFFIXES:
        if s.endswith(salt):
            s = s[: -len(salt)].strip()
    return s


def _normalize_appnum(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.replace(" ", "").upper()


def _cluster_id(canonical_name: str, nda_bla: Optional[str]) -> str:
    key = f"{canonical_name}|{nda_bla or ''}".encode()
    return hashlib.sha1(key).hexdigest()[:12]


def _new_cluster(rec: NormalizedRecord) -> MatchCluster:
    canon = _canonicalize_name(rec.drug_name or rec.generic_name or "")
    cid = _cluster_id(canon, rec.nda_bla_number)
    c = MatchCluster(
        cluster_id=cid,
        canonical_name=canon or (rec.drug_name or rec.generic_name or "unknown"),
        canonical_sponsor=rec.sponsor,
        nda_bla_number=rec.nda_bla_number,
    )
    c.records.append(rec)
    c.sources.add(rec.source)
    return c


def _merge_into(cluster: MatchCluster, rec: NormalizedRecord, score: float) -> None:
    cluster.records.append(rec)
    cluster.sources.add(rec.source)
    # Fill in canonical fields as we learn more.
    if not cluster.canonical_sponsor and rec.sponsor:
        cluster.canonical_sponsor = rec.sponsor
    if not cluster.nda_bla_number and rec.nda_bla_number:
        cluster.nda_bla_number = rec.nda_bla_number
    # Rolling average confidence.
    n = len(cluster.records)
    cluster.match_confidence = (cluster.match_confidence * (n - 1) + score) / n


def _any_indication(c: MatchCluster) -> Optional[str]:
    for r in c.records:
        if r.indication:
            return r.indication
    return None


def _cluster_name_variants(c: MatchCluster) -> list[str]:
    """Every canonicalized name variant the cluster knows about — brand,
    generic, canonical, and any record-level drug/generic names."""
    seen: set[str] = set()
    for v in [c.canonical_name] + [r.drug_name for r in c.records] + [r.generic_name for r in c.records]:
        if not v:
            continue
        cn = _canonicalize_name(v)
        if cn:
            seen.add(cn)
    return list(seen)


def _best_cluster_score(cname: str, c: MatchCluster) -> int:
    """Best fuzzy score between ``cname`` and any known variant of this cluster."""
    best = 0
    for variant in _cluster_name_variants(c):
        s = fuzz.token_set_ratio(cname, variant)
        if s > best:
            best = s
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def cluster_records(records: list[NormalizedRecord], threshold: int = 85) -> list[MatchCluster]:
    # Stage A: exact appnum bucket
    by_appnum: dict[str, MatchCluster] = {}
    remaining: list[NormalizedRecord] = []
    for r in records:
        app_key = _normalize_appnum(r.nda_bla_number)
        if app_key:
            if app_key not in by_appnum:
                by_appnum[app_key] = _new_cluster(r)
            else:
                _merge_into(by_appnum[app_key], r, 100.0)
        else:
            remaining.append(r)

    clusters: list[MatchCluster] = list(by_appnum.values())

    # Stage B: fuzzy collapse by canonical name
    for r in remaining:
        cname = _canonicalize_name(r.drug_name or r.generic_name or "")
        # Also try the generic name separately — bridges brand↔generic across
        # sources (e.g. CT.gov "pembrolizumab" vs openFDA "KEYTRUDA").
        gname = _canonicalize_name(r.generic_name) if r.generic_name else None
        if not cname and not gname:
            clusters.append(_new_cluster(r))
            continue

        best: Optional[MatchCluster] = None
        best_score = 0
        for c in clusters:
            score = _best_cluster_score(cname, c) if cname else 0
            if gname and gname != cname:
                score = max(score, _best_cluster_score(gname, c))
            if score > best_score:
                best_score = score
                best = c

        if best and best_score >= threshold:
            _merge_into(best, r, float(best_score))
        elif best and (threshold - 15) <= best_score < threshold:
            sponsor_score = fuzz.token_set_ratio(r.sponsor or "", best.canonical_sponsor or "")
            indication_score = fuzz.token_set_ratio(r.indication or "", _any_indication(best) or "")
            if sponsor_score >= 85 and indication_score >= 70:
                _merge_into(best, r, float(best_score))
            else:
                clusters.append(_new_cluster(r))
        else:
            clusters.append(_new_cluster(r))

    # Resolve canonical name using precedence: openFDA brand > openFDA generic >
    # CT intervention > PDUFA drug_name.
    for c in clusters:
        _resolve_canonical(c)

    log.info(
        "clustered %d records into %d clusters (multi-source: %d)",
        len(records),
        len(clusters),
        sum(1 for c in clusters if len(c.sources) >= 2),
    )
    return clusters


def _resolve_canonical(c: MatchCluster) -> None:
    openfda_brand = next(
        (r.drug_name for r in c.records if r.source == "openfda" and r.drug_name),
        None,
    )
    openfda_generic = next(
        (r.generic_name for r in c.records if r.source == "openfda" and r.generic_name),
        None,
    )
    ct_name = next(
        (r.drug_name for r in c.records if r.source == "clinicaltrials" and r.drug_name),
        None,
    )
    pdufa_name = next(
        (r.drug_name for r in c.records if r.source == "pdufa" and r.drug_name),
        None,
    )
    chosen = openfda_brand or openfda_generic or ct_name or pdufa_name or c.canonical_name
    if chosen:
        c.canonical_name = chosen
    # Prefer openFDA sponsor if present.
    openfda_sponsor = next(
        (r.sponsor for r in c.records if r.source == "openfda" and r.sponsor),
        None,
    )
    if openfda_sponsor:
        c.canonical_sponsor = openfda_sponsor
