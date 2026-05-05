"""Async fetch layer: ClinicalTrials.gov v2, openFDA drugsfda, FDA PDUFA scrape.

All raw pulls are dumped to ``data/raw_{source}_{YYYY-MM-DD}.json`` for later
inspection and for enabling ``--dry-run`` replay off the most recent dump.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RETRYABLE_EXC = (httpx.HTTPError, httpx.TimeoutException)


def _retry_policy(cfg: dict) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(int(cfg["http"]["max_retries"])),
        wait=wait_exponential(
            multiplier=float(cfg["http"]["backoff_base_seconds"]),
            min=2,
            max=30,
        ),
        retry=retry_if_exception_type(_RETRYABLE_EXC),
        reraise=True,
    )


async def _get_json(client: httpx.AsyncClient, url: str, params: dict, cfg: dict) -> dict:
    async for attempt in _retry_policy(cfg):
        with attempt:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    raise RuntimeError("unreachable")  # pragma: no cover


async def _get_text(client: httpx.AsyncClient, url: str, cfg: dict) -> str:
    async for attempt in _retry_policy(cfg):
        with attempt:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    raise RuntimeError("unreachable")  # pragma: no cover


def _dump_raw(cfg: dict, source: str, payload: Any) -> Path:
    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"raw_{source}_{date.today().isoformat()}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s (%d bytes)", out, out.stat().st_size)
    return out


def _load_most_recent_dump(cfg: dict, source: str) -> list[dict] | None:
    data_dir = Path(cfg["paths"]["data_dir"])
    candidates = sorted(data_dir.glob(f"raw_{source}_*.json"), reverse=True)
    if not candidates:
        return None
    log.info("dry-run: replaying %s", candidates[0])
    return json.loads(candidates[0].read_text())


# ---------------------------------------------------------------------------
# ClinicalTrials.gov v2
# ---------------------------------------------------------------------------


async def fetch_clinical_trials(cfg: dict, lookback_days: int) -> list[dict]:
    """Pull studies matching configured phases/statuses with LastUpdatePostDate
    in the lookback window. Paginates via nextPageToken."""
    src = cfg["sources"]["clinicaltrials"]
    base_url = src["base_url"]
    page_size = int(src["page_size"])
    phases = src["phases"]
    statuses = src["statuses"]

    today = date.today()
    start = (today - timedelta(days=lookback_days)).isoformat()
    end = today.isoformat()

    phase_expr = " OR ".join(phases)
    query_term = (
        f"AREA[Phase]({phase_expr}) "
        f"AND AREA[LastUpdatePostDate]RANGE[{start},{end}]"
    )

    params = {
        "query.term": query_term,
        "filter.overallStatus": "|".join(statuses),
        "pageSize": page_size,
        "format": "json",
    }

    studies: list[dict] = []
    timeout = httpx.Timeout(float(cfg["http"]["timeout_seconds"]))
    headers = {"User-Agent": cfg["http"]["user_agent"]}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        page_token: str | None = None
        page = 0
        while True:
            page += 1
            if page_token:
                params["pageToken"] = page_token
            else:
                params.pop("pageToken", None)

            try:
                data = await _get_json(client, base_url, params, cfg)
            except _RETRYABLE_EXC as e:
                log.error("clinicaltrials fetch failed on page %d: %s", page, e)
                raise

            batch = data.get("studies", []) or []
            studies.extend(batch)
            log.info("clinicaltrials page %d: %d studies (total %d)", page, len(batch), len(studies))

            page_token = data.get("nextPageToken")
            if not page_token or not batch:
                break

    return studies


# ---------------------------------------------------------------------------
# Per-drug ClinicalTrials.gov enrichment
# ---------------------------------------------------------------------------
#
# After the event-driven date-filtered fetch above, we re-query CT.gov scoped
# to each drug name (no date filter) to recover that drug's full trial portfolio
# — the wide indication coverage that commercial pharma trackers show. Cached
# per-drug on disk so we aren't re-pulling Keytruda's 600-study history weekly.


def _drug_slug(name: str) -> str:
    """Filesystem-safe slug for cache filenames (lowercase, alnum + dash)."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "unknown"


def _drug_cache_path(cfg: dict, drug: str) -> Path:
    data_dir = Path(cfg["paths"]["data_dir"])
    cache_dir = data_dir / "drug_trials_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_drug_slug(drug)}.json"


def _load_drug_cache(cfg: dict, drug: str, ttl_days: int) -> list[dict] | None:
    """Return cached studies if the cache file is fresh enough, else None."""
    path = _drug_cache_path(cfg, drug)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"]).date()
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    if (date.today() - fetched_at).days > ttl_days:
        return None
    return payload.get("studies") or []


def _save_drug_cache(cfg: dict, drug: str, studies: list[dict]) -> None:
    path = _drug_cache_path(cfg, drug)
    payload = {
        "drug": drug,
        "fetched_at": date.today().isoformat(),
        "study_count": len(studies),
        "studies": studies,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


async def fetch_trials_for_drug(
    client: httpx.AsyncClient,
    drug: str,
    cfg: dict,
    max_studies: int,
) -> list[dict]:
    """Query CT.gov v2 by intervention name with NO date filter.

    Returns up to *max_studies* studies for the drug, paginated. This is the
    per-drug counterpart to the date-windowed ``fetch_clinical_trials``: it's
    used to fill in the full trial portfolio (and therefore the full indication
    list) for drugs we already know about from event-driven sources.
    """
    src = cfg["sources"]["clinicaltrials"]
    base_url = src["base_url"]
    page_size = int(src["page_size"])

    params = {
        "query.intr": drug,
        "pageSize": page_size,
        "format": "json",
    }

    studies: list[dict] = []
    page_token: str | None = None
    page = 0
    while True:
        page += 1
        if page_token:
            params["pageToken"] = page_token
        else:
            params.pop("pageToken", None)

        try:
            data = await _get_json(client, base_url, params, cfg)
        except _RETRYABLE_EXC as e:
            log.warning("per-drug ct fetch failed for %r on page %d: %s", drug, page, e)
            break

        batch = data.get("studies", []) or []
        studies.extend(batch)
        page_token = data.get("nextPageToken")
        if not page_token or not batch or len(studies) >= max_studies:
            break

    return studies[:max_studies]


def _extract_enrichment_drug_names(
    ct_studies: list[dict],
    openfda_results: list[dict],
    cap: int,
) -> list[str]:
    """Build the de-duplicated list of drug names worth fetching full corpora for.

    Source priority — openFDA brand > openFDA generic > CT.gov interventions —
    so when several names canonicalize to the same drug we keep the most
    "official" form for the API query.
    """
    # Local import to avoid widening this module's import surface for one helper.
    from .match import _canonicalize_name

    seen_canonical: set[str] = set()
    ordered: list[str] = []

    def _add(name: str | None) -> None:
        if not name:
            return
        canonical = _canonicalize_name(name)
        if not canonical or canonical in seen_canonical:
            return
        seen_canonical.add(canonical)
        ordered.append(name.strip())

    # openFDA brand_name first (most "official" identifier)
    for result in openfda_results:
        for product in result.get("products") or []:
            _add(product.get("brand_name"))
    # openFDA generic / active ingredient
    for result in openfda_results:
        for product in result.get("products") or []:
            for ai in product.get("active_ingredients") or []:
                _add(ai.get("name"))
    # CT.gov interventions (drug + biological)
    for study in ct_studies:
        ps = study.get("protocolSection") or {}
        arms = ps.get("armsInterventionsModule") or {}
        for interv in arms.get("interventions") or []:
            if str(interv.get("type", "")).upper() in {"DRUG", "BIOLOGICAL"}:
                _add(interv.get("name"))

    if cap and len(ordered) > cap:
        log.info(
            "per-drug enrichment: capping %d candidate drugs at max_drugs=%d",
            len(ordered),
            cap,
        )
        ordered = ordered[:cap]
    return ordered


async def _enrich_with_per_drug_trials(
    ct_studies: list[dict],
    openfda_results: list[dict],
    cfg: dict,
) -> list[dict]:
    """Run per-drug CT.gov fetches for every drug in this run, merge into ct_studies.

    Dedupes the final list by NCT ID. Honours cache TTL, max_drugs, and
    max_studies_per_drug knobs from config.yaml.
    """
    enrich_cfg = (cfg["sources"]["clinicaltrials"] or {}).get("per_drug_enrichment") or {}
    if not enrich_cfg.get("enabled", True):
        log.info("per-drug enrichment disabled in config — skipping")
        return ct_studies

    max_drugs = int(enrich_cfg.get("max_drugs", 200))
    max_per_drug = int(enrich_cfg.get("max_studies_per_drug", 500))
    ttl_days = int(enrich_cfg.get("cache_ttl_days", 7))
    pace = float(enrich_cfg.get("pace_seconds", 0.2))

    drugs = _extract_enrichment_drug_names(ct_studies, openfda_results, max_drugs)
    if not drugs:
        log.info("per-drug enrichment: no drug names extracted — skipping")
        return ct_studies

    log.info(
        "per-drug enrichment: fetching CT.gov corpus for %d drugs (cache_ttl=%dd, max_per_drug=%d)",
        len(drugs),
        ttl_days,
        max_per_drug,
    )

    timeout = httpx.Timeout(float(cfg["http"]["timeout_seconds"]))
    headers = {"User-Agent": cfg["http"]["user_agent"]}

    cache_hits = 0
    fetched_drugs = 0
    fetched_studies_total = 0
    new_studies: list[dict] = []

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for drug in drugs:
            cached = _load_drug_cache(cfg, drug, ttl_days)
            if cached is not None:
                cache_hits += 1
                new_studies.extend(cached)
                continue
            try:
                studies = await fetch_trials_for_drug(client, drug, cfg, max_per_drug)
            except Exception as e:  # noqa: BLE001
                log.warning("per-drug fetch failed for %r: %s — continuing", drug, e)
                continue
            fetched_drugs += 1
            fetched_studies_total += len(studies)
            _save_drug_cache(cfg, drug, studies)
            new_studies.extend(studies)
            if pace:
                await asyncio.sleep(pace)

    # Merge: dedupe by NCT ID, preferring the existing event-driven study
    # (it has the freshest LastUpdatePostDate-driven status).
    by_nct: dict[str, dict] = {}
    extras: list[dict] = []  # studies with no NCT ID (rare)
    for study in ct_studies:
        nct = ((study.get("protocolSection") or {}).get("identificationModule") or {}).get("nctId")
        if nct:
            by_nct[nct] = study
        else:
            extras.append(study)
    pre_count = len(by_nct) + len(extras)
    added = 0
    for study in new_studies:
        nct = ((study.get("protocolSection") or {}).get("identificationModule") or {}).get("nctId")
        if not nct:
            extras.append(study)
            added += 1
            continue
        if nct not in by_nct:
            by_nct[nct] = study
            added += 1

    merged = list(by_nct.values()) + extras
    log.info(
        "per-drug enrichment done: %d cache_hits, %d drugs fetched (%d studies), "
        "merged %d → %d ct studies (+%d new)",
        cache_hits,
        fetched_drugs,
        fetched_studies_total,
        pre_count,
        len(merged),
        added,
    )
    return merged


# ---------------------------------------------------------------------------
# openFDA drugsfda
# ---------------------------------------------------------------------------


async def fetch_openfda_drugsfda(cfg: dict, lookback_days: int) -> list[dict]:
    """Pull drugsfda applications whose submissions changed status (AP or TA)
    within the lookback window."""
    src = cfg["sources"]["openfda"]
    base_url = src["base_url"]
    page_size = int(src["page_size"])

    today = date.today()
    start = (today - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    # openFDA uses Lucene-style ranges; brackets are inclusive.
    # Use spaces around TO so httpx encodes them as '+' (Lucene's expected form);
    # literal '+' would be re-encoded as %2B and the server returns 500.
    search = f"submissions.submission_status_date:[{start} TO {end}]"

    results: list[dict] = []
    timeout = httpx.Timeout(float(cfg["http"]["timeout_seconds"]))
    headers = {"User-Agent": cfg["http"]["user_agent"]}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        skip = 0
        page = 0
        while True:
            page += 1
            params = {"search": search, "limit": page_size, "skip": skip}
            try:
                data = await _get_json(client, base_url, params, cfg)
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 404:
                    # openFDA returns 404 when no results match.
                    log.info("openfda page %d: 0 results (404 = empty search)", page)
                    break
                raise

            batch = data.get("results", []) or []
            results.extend(batch)
            meta = data.get("meta", {}).get("results", {})
            total = meta.get("total", len(results))
            log.info(
                "openfda page %d: %d results (cum %d of %d)",
                page,
                len(batch),
                len(results),
                total,
            )

            if not batch or len(results) >= total:
                break
            skip += page_size
            # Defensive pacing well under the 240/min public limit.
            await asyncio.sleep(0.25)

    return results


# ---------------------------------------------------------------------------
# FDA PDUFA page scrape
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "drug_name": [
        "drug", "proprietary", "brand", "tradename", "product", "catalyst",
        "drug name", "compound",
    ],
    "generic_name": [
        "generic", "established", "active ingredient", "nonproprietary",
        "drug class", "generic name", "molecule",
    ],
    "applicant": ["applicant", "sponsor", "company", "ticker", "firm"],
    "nda_bla_number": ["application", "nda", "bla", "app number", "application number"],
    "submission_type": ["type", "submission", "stage", "action type"],
    "pdufa_date": [
        "pdufa", "target action", "goal date", "action date", "catalyst date",
        "date", "pdufa date", "fda date", "decision date", "target date",
    ],
    "indication": ["indication", "proposed indication", "use", "disease", "therapeutic area"],
    "submission_date": ["submission date", "received", "filed"],
}


# Browser-like headers for third-party calendar scrapes (e.g. BioPharmCatalyst),
# which reject the config's plain bot User-Agent. Only used by the PDUFA/approval
# HTML scrapers — openFDA and ClinicalTrials.gov still use the config UA.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _slugify(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _map_columns(headers: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, h in enumerate(headers):
        h_slug = _slugify(h)
        if not h_slug:
            continue
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in mapping:
                continue
            if any(alias in h_slug for alias in aliases):
                mapping[canonical] = idx
                break
    return mapping


def _detect_header_row(table) -> Any:
    # Prefer a <tr> containing <th>; fall back to first <tr> containing a known keyword.
    for tr in table.find_all("tr"):
        if tr.find("th"):
            return tr
    keywords = {"drug", "pdufa", "target action", "applicant", "nda", "bla"}
    for tr in table.find_all("tr"):
        text = _slugify(tr.get_text(" ", strip=True))
        if any(k in text for k in keywords):
            return tr
    return None


def _parse_pdufa_tables(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for table in soup.find_all("table"):
        header_row = _detect_header_row(table)
        if not header_row:
            continue
        header_cells = header_row.find_all(["th", "td"])
        headers = [c.get_text(" ", strip=True) for c in header_cells]
        col_map = _map_columns(headers)
        if "drug_name" not in col_map or "pdufa_date" not in col_map:
            continue

        for tr in table.find_all("tr"):
            if tr is header_row:
                continue
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            row: dict[str, Any] = {}
            for key, idx in col_map.items():
                if idx < len(cells):
                    val = cells[idx]
                    if val:
                        row[key] = val
            if row.get("drug_name") and row.get("pdufa_date"):
                rows.append(row)
    return rows


def _dump_html_on_parse_failure(cfg: dict, html: str, label: str) -> Path | None:
    """Save raw HTML bytes to disk for offline diagnosis when the table parser
    returns zero rows. Distinct from ``_dump_raw`` (which writes parsed dicts)."""
    try:
        data_dir = Path(cfg["paths"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        out = data_dir / f"raw_{label}_html_{date.today().isoformat()}.html"
        out.write_text(html, encoding="utf-8")
        return out
    except Exception as e:  # pragma: no cover - defensive
        log.warning("failed to save raw %s HTML: %s", label, e)
        return None


async def fetch_pdufa_page(cfg: dict) -> list[dict]:
    url = cfg["sources"]["pdufa"]["pdufa_url"]
    timeout = httpx.Timeout(float(cfg["http"]["timeout_seconds"]))
    async with httpx.AsyncClient(
        timeout=timeout, headers=_BROWSER_HEADERS, follow_redirects=True
    ) as client:
        try:
            html = await _get_text(client, url, cfg)
        except _RETRYABLE_EXC as e:
            log.warning("PDUFA page fetch failed: %s", e)
            return []
    rows = _parse_pdufa_tables(html)
    log.info("pdufa page parsed: %d rows", len(rows))
    if not rows:
        dump_path = _dump_html_on_parse_failure(cfg, html, "pdufa")
        log.warning(
            "pdufa page returned %d bytes of HTML but 0 parseable rows — "
            "table structure may have changed. Raw HTML saved to %s",
            len(html),
            dump_path,
        )
    return rows


async def fetch_approvals_page(cfg: dict) -> list[dict]:
    url = cfg["sources"]["pdufa"]["approvals_url"]
    timeout = httpx.Timeout(float(cfg["http"]["timeout_seconds"]))
    async with httpx.AsyncClient(
        timeout=timeout, headers=_BROWSER_HEADERS, follow_redirects=True
    ) as client:
        try:
            html = await _get_text(client, url, cfg)
        except _RETRYABLE_EXC as e:
            log.warning("FDA approvals page fetch failed: %s", e)
            return []
    rows = _parse_pdufa_tables(html)
    log.info("approvals page parsed: %d rows", len(rows))
    if not rows:
        dump_path = _dump_html_on_parse_failure(cfg, html, "approvals")
        log.warning(
            "approvals page returned %d bytes of HTML but 0 parseable rows — "
            "table structure may have changed. Raw HTML saved to %s",
            len(html),
            dump_path,
        )
    return rows


async def fetch_fda_press_rss(cfg: dict) -> list[dict]:
    """Minimal RSS fallback — return items whose title mentions PDUFA/approves."""
    url = cfg["sources"]["pdufa"]["rss_fallback_url"]
    timeout = httpx.Timeout(float(cfg["http"]["timeout_seconds"]))
    headers = {"User-Agent": cfg["http"]["user_agent"]}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            text = await _get_text(client, url, cfg)
    except _RETRYABLE_EXC as e:
        log.warning("FDA press RSS fetch failed: %s", e)
        return []

    items: list[dict] = []
    try:
        soup = BeautifulSoup(text, "xml")
        for item in soup.find_all("item"):
            title = (item.title.text if item.title else "") or ""
            if not re.search(r"pdufa|approve", title, re.IGNORECASE):
                continue
            items.append(
                {
                    "drug_name": title,
                    "applicant": None,
                    "pdufa_date": None,
                    "source": "rss",
                    "link": item.link.text if item.link else None,
                    "pubDate": item.pubDate.text if item.pubDate else None,
                }
            )
    except Exception as e:  # pragma: no cover - defensive
        log.warning("RSS parse failed: %s", e)
    log.info("rss fallback parsed: %d items", len(items))
    return items


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def fetch_all(cfg: dict, dry_run: bool = False) -> dict:
    """Sync entry point that orchestrates all sources concurrently."""
    return asyncio.run(_fetch_all_async(cfg, dry_run=dry_run))


async def _fetch_all_async(cfg: dict, dry_run: bool = False) -> dict:
    warnings: list[str] = []
    lookback = int(cfg["lookback_days"])

    if dry_run:
        ct = _load_most_recent_dump(cfg, "clinicaltrials") or []
        of = _load_most_recent_dump(cfg, "openfda") or []
        pdufa_rows = _load_most_recent_dump(cfg, "pdufa") or []
        if not (ct and of and pdufa_rows):
            warnings.append("dry-run: one or more cached dumps missing")
        return {"ct": ct, "openfda": of, "pdufa": pdufa_rows, "warnings": warnings}

    async def _safe(coro, name: str) -> list[dict]:
        try:
            return await coro
        except Exception as e:  # noqa: BLE001
            log.error("%s fetch failed: %s", name, e)
            warnings.append(f"{name} fetch failed: {e}")
            return []

    ct_task = _safe(fetch_clinical_trials(cfg, lookback), "clinicaltrials")
    of_task = _safe(fetch_openfda_drugsfda(cfg, lookback), "openfda")
    pdufa_task = _safe(fetch_pdufa_page(cfg), "pdufa")

    ct, openfda_results, pdufa_rows = await asyncio.gather(ct_task, of_task, pdufa_task)

    # Per-drug enrichment: re-query CT.gov by drug name (no date filter) for
    # every drug surfaced in this run. This is what brings in the full
    # indication portfolio per drug. Failures here are non-fatal — we already
    # have the date-windowed studies above.
    try:
        ct = await _enrich_with_per_drug_trials(ct, openfda_results, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("per-drug enrichment failed wholesale: %s — using event-driven CT only", e)
        warnings.append(f"per-drug enrichment failed: {e}")

    _dump_raw(cfg, "clinicaltrials", ct)
    _dump_raw(cfg, "openfda", openfda_results)
    _dump_raw(cfg, "pdufa", pdufa_rows)

    return {
        "ct": ct,
        "openfda": openfda_results,
        "pdufa": pdufa_rows,
        "warnings": warnings,
    }
