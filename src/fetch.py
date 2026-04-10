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
    "drug_name": ["drug", "proprietary", "brand", "tradename", "product", "catalyst"],
    "generic_name": [
        "generic",
        "established",
        "active ingredient",
        "nonproprietary",
        "drug class",
    ],
    "applicant": ["applicant", "sponsor", "company", "ticker"],
    "nda_bla_number": ["application", "nda", "bla", "app number"],
    "submission_type": ["type", "submission", "stage"],
    "pdufa_date": [
        "pdufa",
        "target action",
        "goal date",
        "action date",
        "catalyst date",
        "date",
    ],
    "indication": ["indication", "proposed indication", "use", "disease"],
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
        log.warning(
            "pdufa page returned %d bytes of HTML but 0 parseable rows — "
            "table structure may have changed. Raw HTML dumped to data/raw_pdufa_*.json",
            len(html),
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
    pdufa_task = _safe(fetch_pdufa_page(cfg), "pdufa_page")

    ct, openfda_results, pdufa_rows = await asyncio.gather(ct_task, of_task, pdufa_task)

    # Fallback chain for PDUFA if primary page returned nothing.
    if not pdufa_rows:
        warnings.append("PDUFA primary scrape returned 0 rows; trying approvals page")
        pdufa_rows = await _safe(fetch_approvals_page(cfg), "pdufa_approvals")
    if not pdufa_rows:
        warnings.append("PDUFA approvals page returned 0 rows; trying RSS")
        pdufa_rows = await _safe(fetch_fda_press_rss(cfg), "pdufa_rss")
    if not pdufa_rows:
        warnings.append("PDUFA data may be stale — all scrape sources returned 0 rows")

    _dump_raw(cfg, "clinicaltrials", ct)
    _dump_raw(cfg, "openfda", openfda_results)
    _dump_raw(cfg, "pdufa", pdufa_rows)

    return {
        "ct": ct,
        "openfda": openfda_results,
        "pdufa": pdufa_rows,
        "warnings": warnings,
    }
