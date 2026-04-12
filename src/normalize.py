"""Source-specific → NormalizedRecord mapping.

All dates coerce to ISO ``YYYY-MM-DD``. Missing fields remain ``None``
(never ``""`` or ``"N/A"``). Sponsor names are title-cased; NDA/BLA numbers
are uppercased.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from dateutil import parser as dateparser

log = logging.getLogger(__name__)


@dataclass
class NormalizedRecord:
    source: str  # "clinicaltrials" | "openfda" | "pdufa"
    source_id: str
    drug_name: Optional[str] = None
    generic_name: Optional[str] = None
    sponsor: Optional[str] = None
    phase: Optional[str] = None  # "PHASE1" | "PHASE2" | "PHASE3"
    status: Optional[str] = None
    approval_date: Optional[str] = None
    pdufa_date: Optional[str] = None
    pdufa_status: Optional[str] = None
    nct_id: Optional[str] = None
    nda_bla_number: Optional[str] = None
    submission_type: Optional[str] = None
    therapeutic_area: Optional[str] = None
    indication: Optional[str] = None
    completion_date: Optional[str] = None
    results_posted: Optional[bool] = None
    signal_type: Optional[str] = None
    raw_ref: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(value: Any) -> Optional[str]:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, date):
        return value.isoformat()
    try:
        dt = dateparser.parse(str(value), default=None, fuzzy=True)
        return dt.date().isoformat() if dt else None
    except (ValueError, TypeError, OverflowError):
        return None


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _title(value: Any) -> Optional[str]:
    s = _clean(value)
    if not s:
        return None
    # Keep acronyms in all caps if already so.
    return s.title() if s.isupper() or s.islower() else s


def _normalize_appnum(value: Any) -> Optional[str]:
    s = _clean(value)
    if not s:
        return None
    compact = s.replace(" ", "").upper()
    # openFDA application numbers come as "NDA212345" or just "212345".
    if compact.isdigit():
        return compact
    return compact


_THERAPEUTIC_KEYWORDS: dict[str, list[str]] = {
    "oncology": [
        "cancer",
        "carcinoma",
        "tumor",
        "tumour",
        "lymphoma",
        "leukemia",
        "leukaemia",
        "melanoma",
        "sarcoma",
        "myeloma",
        "neoplasm",
        "glioma",
    ],
    "neurology": [
        "alzheimer",
        "parkinson",
        "epilepsy",
        "multiple sclerosis",
        "als",
        "migraine",
        "neurodegenerative",
        "stroke",
        "seizure",
    ],
    "cardiology": [
        "cardio",
        "heart failure",
        "hypertension",
        "atrial",
        "myocardial",
        "coronary",
    ],
    "nephrology": ["kidney", "renal", "nephrotic", "dialysis", "ckd"],
    "rare disease": ["rare disease", "orphan", "ultra-rare"],
    "metabolic": ["diabetes", "obesity", "metabolic", "nash", "nafld"],
    "infectious disease": ["hiv", "hepatitis", "tuberculosis", "covid", "sars-cov", "bacterial", "viral"],
    "immunology": ["lupus", "psoriasis", "rheumatoid", "crohn", "colitis", "autoimmune"],
    "dermatology": ["dermatitis", "eczema", "psoriasis", "acne"],
    "ophthalmology": ["macular", "retina", "glaucoma", "dry eye"],
    "respiratory": ["asthma", "copd", "pulmonary"],
}


def _derive_therapeutic_area(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for area, keywords in _THERAPEUTIC_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return area
    return None


# ---------------------------------------------------------------------------
# ClinicalTrials.gov v2
# ---------------------------------------------------------------------------


def normalize_clinical_trial(study: dict) -> Optional[NormalizedRecord]:
    """Map a ClinicalTrials.gov v2 study to a NormalizedRecord.

    v2 payload shape: ``study.protocolSection.{identificationModule, statusModule,
    sponsorCollaboratorsModule, designModule, conditionsModule, armsInterventionsModule}``
    """
    ps = study.get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    status_mod = ps.get("statusModule") or {}
    sponsor_mod = ps.get("sponsorCollaboratorsModule") or {}
    design = ps.get("designModule") or {}
    conditions_mod = ps.get("conditionsModule") or {}
    arms_mod = ps.get("armsInterventionsModule") or {}
    results_section = study.get("resultsSection") or {}

    nct_id = _clean(ident.get("nctId"))
    if not nct_id:
        return None

    # Prefer first drug intervention name; fall back to study brief title.
    drug_name: Optional[str] = None
    for interv in arms_mod.get("interventions", []) or []:
        if interv.get("type", "").upper() in {"DRUG", "BIOLOGICAL"}:
            drug_name = _clean(interv.get("name"))
            if drug_name:
                break
    if not drug_name:
        drug_name = _clean(ident.get("briefTitle"))

    sponsor = _clean((sponsor_mod.get("leadSponsor") or {}).get("name"))

    phases = design.get("phases") or []
    # Pick highest phase (1 < 2 < 3).
    phase: Optional[str] = None
    ranking = {"PHASE1": 1, "PHASE2": 2, "PHASE3": 3}
    for p in phases:
        p_up = str(p).upper().replace(" ", "")
        if p_up in ranking and (phase is None or ranking[p_up] > ranking[phase]):
            phase = p_up

    status = _clean(status_mod.get("overallStatus"))
    completion = _parse_date(
        (status_mod.get("completionDateStruct") or {}).get("date")
        or (status_mod.get("primaryCompletionDateStruct") or {}).get("date")
    )

    conditions = conditions_mod.get("conditions") or []
    indication = ", ".join(c for c in conditions if c) if conditions else None

    results_posted = bool(results_section) if results_section else None

    return NormalizedRecord(
        source="clinicaltrials",
        source_id=nct_id,
        drug_name=drug_name,
        sponsor=sponsor,
        phase=phase,
        status=status,
        nct_id=nct_id,
        indication=indication,
        completion_date=completion,
        results_posted=results_posted,
        therapeutic_area=_derive_therapeutic_area(indication),
        raw_ref={"nct_id": nct_id},
    )


# ---------------------------------------------------------------------------
# openFDA drugsfda
# ---------------------------------------------------------------------------


def normalize_openfda(result: dict, lookback_start: str | None = None) -> list[NormalizedRecord]:
    """One openFDA application may carry multiple submissions; emit a record per
    approval (submission_status ``AP`` or ``TA``) within the lookback window.

    *lookback_start* is an ISO date string (``YYYY-MM-DD``). Submissions whose
    ``submission_status_date`` predates it are skipped — they're old approvals
    on an application that was merely updated recently, not new actions.
    """
    out: list[NormalizedRecord] = []
    app_num_raw = result.get("application_number")
    app_num = _normalize_appnum(app_num_raw)
    sponsor = _title(result.get("sponsor_name"))

    products = result.get("products") or []
    brand = None
    generic = None
    dosage_forms: list[str] = []
    pharmacologic_classes: list[str] = []
    for p in products:
        if not brand:
            brand = _clean(p.get("brand_name"))
        if not generic:
            generic = _clean(p.get("active_ingredients", [{}])[0].get("name") if p.get("active_ingredients") else None)
        if p.get("dosage_form"):
            dosage_forms.append(str(p["dosage_form"]))
        if isinstance(p.get("pharmacologic_class"), list):
            pharmacologic_classes.extend(p["pharmacologic_class"])

    submission_type = None
    if app_num_raw:
        if str(app_num_raw).upper().startswith("NDA"):
            submission_type = "NDA"
        elif str(app_num_raw).upper().startswith("BLA"):
            submission_type = "BLA"
        elif str(app_num_raw).upper().startswith("ANDA"):
            submission_type = "ANDA"

    therapeutic_area = _derive_therapeutic_area(
        " ".join(pharmacologic_classes + dosage_forms) if pharmacologic_classes or dosage_forms else None
    )

    for sub in result.get("submissions") or []:
        status_code = _clean(sub.get("submission_status"))
        if status_code not in {"AP", "TA"}:
            continue
        approval_date = _parse_date(sub.get("submission_status_date"))
        if lookback_start and approval_date and approval_date < lookback_start:
            continue
        sub_type = _clean(sub.get("submission_type")) or submission_type
        rec = NormalizedRecord(
            source="openfda",
            source_id=f"{app_num or 'UNKNOWN'}:{sub.get('submission_number', '')}",
            drug_name=brand or generic,
            generic_name=generic,
            sponsor=sponsor,
            status=status_code,
            approval_date=approval_date,
            nda_bla_number=app_num,
            submission_type=sub_type,
            therapeutic_area=therapeutic_area,
            raw_ref={"application_number": app_num_raw, "submission_number": sub.get("submission_number")},
        )
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# PDUFA scrape rows
# ---------------------------------------------------------------------------


def normalize_pdufa(row: dict) -> Optional[NormalizedRecord]:
    drug_name = _clean(row.get("drug_name"))
    pdufa_date = _parse_date(row.get("pdufa_date"))
    if not drug_name or not pdufa_date:
        return None

    app_num = _normalize_appnum(row.get("nda_bla_number"))
    submission_type = None
    if app_num:
        if app_num.startswith("NDA"):
            submission_type = "NDA"
        elif app_num.startswith("BLA"):
            submission_type = "BLA"
    if not submission_type:
        submission_type = _clean(row.get("submission_type"))

    sponsor = _title(row.get("applicant"))
    indication = _clean(row.get("indication"))

    return NormalizedRecord(
        source="pdufa",
        source_id=f"{drug_name}|{app_num or ''}|{pdufa_date}",
        drug_name=drug_name,
        generic_name=_clean(row.get("generic_name")),
        sponsor=sponsor,
        pdufa_date=pdufa_date,
        nda_bla_number=app_num,
        submission_type=submission_type,
        indication=indication,
        therapeutic_area=_derive_therapeutic_area(indication),
        raw_ref=dict(row),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def normalize_all(raw: dict, lookback_days: int | None = None) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    lookback_start: str | None = None
    if lookback_days:
        lookback_start = (date.today() - timedelta(days=lookback_days)).isoformat()

    for study in raw.get("ct", []) or []:
        try:
            rec = normalize_clinical_trial(study)
        except Exception as e:  # noqa: BLE001
            log.warning("normalize_clinical_trial failed: %s", e)
            continue
        if rec:
            records.append(rec)

    openfda_raw_count = 0
    for result in raw.get("openfda", []) or []:
        try:
            recs = normalize_openfda(result, lookback_start=lookback_start)
        except Exception as e:  # noqa: BLE001
            log.warning("normalize_openfda failed: %s", e)
            continue
        openfda_raw_count += len(result.get("submissions", []) or [])
        records.extend(recs)

    for row in raw.get("pdufa", []) or []:
        try:
            rec = normalize_pdufa(row)
        except Exception as e:  # noqa: BLE001
            log.warning("normalize_pdufa failed: %s", e)
            continue
        if rec:
            records.append(rec)

    openfda_kept = sum(1 for r in records if r.source == "openfda")
    log.info(
        "normalized %d records (ct=%d, openfda=%d, pdufa=%d)",
        len(records),
        sum(1 for r in records if r.source == "clinicaltrials"),
        openfda_kept,
        sum(1 for r in records if r.source == "pdufa"),
    )
    if lookback_start and openfda_raw_count:
        log.info(
            "openfda date filter: %d total submissions → %d within lookback window (since %s)",
            openfda_raw_count,
            openfda_kept,
            lookback_start,
        )
    return records
