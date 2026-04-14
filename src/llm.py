"""Claude API wrapper for content generation.

Generates narrative text (newsletter sections, LinkedIn posts) from
structured pipeline data. Reads ANTHROPIC_API_KEY from the environment.
Gracefully degrades if the key is missing or the API call fails.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

_MODEL_DEFAULT = "claude-sonnet-4-20250514"


def _get_client(cfg: dict, channel: str = "llm"):
    """Return an Anthropic client or None if the SDK/key is unavailable.

    The *channel* argument names the caller (e.g. "newsletter", "linkedin")
    so the warning line makes it obvious which output is degrading.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning(
            "%s: ANTHROPIC_API_KEY not set — skipping LLM call, writing fallback output",
            channel,
        )
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        log.warning("%s: anthropic package not installed — run: pip install anthropic", channel)
        return None


def _model(cfg: dict) -> str:
    return cfg.get("content", {}).get("llm", {}).get("model", _MODEL_DEFAULT)


def _call(client, model: str, system: str, user: str) -> str | None:
    """Make a single Claude API call. Returns the text response or None."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    except Exception as e:  # noqa: BLE001
        log.error("Claude API call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Newsletter narrative
# ---------------------------------------------------------------------------

_NEWSLETTER_SYSTEM = """\
You are a pharmaceutical industry analyst writing a weekly FDA intelligence \
digest for healthcare professionals and biotech investors.

Rules:
- Write in a professional but accessible tone.
- Be specific about drug names, sponsors, and therapeutic areas.
- Do not speculate beyond what the data shows.
- If a section has no data, write a brief note saying so.
- Return valid JSON with exactly these keys: \
"opening", "approvals", "pipeline", "watch"."""


def generate_newsletter_narrative(report: dict, cfg: dict) -> dict[str, str] | None:
    """Generate narrative sections for the newsletter.

    Returns ``{"opening": "...", "approvals": "...", "pipeline": "...", "watch": "..."}``
    or ``None`` if the API is unavailable.
    """
    client = _get_client(cfg, channel="newsletter")
    if not client:
        return None

    # Build a compact data summary for the prompt
    run_date = report.get("run_date", "unknown")
    source_counts = report.get("source_counts", {})
    clusters = report.get("clusters", [])

    new_approvals = []
    high_signal = []
    status_changes = []
    for c in clusters:
        flags = c.get("flags", {})
        if flags.get("high_signal"):
            high_signal.append({
                "drug": c.get("canonical_name"),
                "sponsor": c.get("canonical_sponsor"),
                "sources": c.get("sources", []),
            })
        for rec in c.get("records", []):
            if rec.get("signal_type") == "new_approval":
                new_approvals.append({
                    "drug": rec.get("drug_name"),
                    "sponsor": rec.get("sponsor"),
                    "approval_date": rec.get("approval_date"),
                    "nda_bla": rec.get("nda_bla_number"),
                    "area": rec.get("therapeutic_area"),
                })
            if rec.get("signal_type") in ("trial_complete", "trial_terminated", "results_posted"):
                status_changes.append({
                    "drug": rec.get("drug_name"),
                    "sponsor": rec.get("sponsor"),
                    "phase": rec.get("phase"),
                    "signal": rec.get("signal_type"),
                    "nct": rec.get("nct_id"),
                })

    data_summary = json.dumps({
        "run_date": run_date,
        "source_counts": source_counts,
        "total_clusters": len(clusters),
        "new_approvals": new_approvals[:20],
        "status_changes": status_changes[:20],
        "high_signal_clusters": high_signal[:15],
    }, indent=2)

    user_prompt = f"""\
Based on the following FDA pipeline data from {run_date}, write four sections:

1. OPENING (2-3 sentences): Highlight the most newsworthy development this week.
2. KEY APPROVALS: Narrative paragraph covering new FDA approvals. If none, note that.
3. PIPELINE HIGHLIGHTS: Notable trial status changes — completions, terminations, results posted.
4. WHAT TO WATCH: High-signal clusters (drugs appearing in multiple data sources) and upcoming catalysts.

Data:
{data_summary}

Return ONLY valid JSON: {{"opening": "...", "approvals": "...", "pipeline": "...", "watch": "..."}}"""

    raw = _call(client, _model(cfg), _NEWSLETTER_SYSTEM, user_prompt)
    if not raw:
        return None

    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        log.warning("failed to parse newsletter LLM response: %s", e)
        log.debug("raw response: %s", raw[:500])
        return None


# ---------------------------------------------------------------------------
# LinkedIn posts
# ---------------------------------------------------------------------------

_LINKEDIN_SYSTEM = """\
You are a healthcare/biotech content strategist writing LinkedIn posts.

Rules:
- Professional tone, 150-250 words each.
- Include 3-5 relevant hashtags at the end (e.g. #FDA #BiotechNews #DrugApprovals).
- End the post body (before hashtags) with a question or insight to drive engagement.
- Do not use emojis.
- Return valid JSON array: [{"event_type": "...", "drug": "...", "text": "...", "hashtags": ["..."]}]"""


def generate_linkedin_posts(report: dict, cfg: dict) -> list[dict] | None:
    """Generate LinkedIn posts for notable pipeline events.

    Returns a list of post dicts, ``[]`` if no events match the filter, or
    ``None`` if the API/SDK is unavailable. Callers use the distinction to
    explain the output to the user.
    """
    client = _get_client(cfg, channel="linkedin")
    if not client:
        return None

    max_posts = cfg.get("content", {}).get("linkedin", {}).get("max_posts", 5)

    # Select notable events to post about
    events = []
    for c in report.get("clusters", []):
        for rec in c.get("records", []):
            if rec.get("signal_type") == "new_approval":
                events.append({
                    "event_type": "new_approval",
                    "drug": rec.get("drug_name"),
                    "sponsor": rec.get("sponsor"),
                    "approval_date": rec.get("approval_date"),
                    "nda_bla": rec.get("nda_bla_number"),
                    "therapeutic_area": rec.get("therapeutic_area"),
                    "indication": rec.get("indication"),
                })
            elif rec.get("signal_type") == "trial_complete" and rec.get("phase") == "PHASE3":
                events.append({
                    "event_type": "phase3_complete",
                    "drug": rec.get("drug_name"),
                    "sponsor": rec.get("sponsor"),
                    "nct": rec.get("nct_id"),
                    "indication": rec.get("indication"),
                })

    log.info(
        "linkedin: %d candidate events from %d clusters (approvals=%d, phase3_complete=%d)",
        len(events),
        len(report.get("clusters", [])),
        sum(1 for e in events if e["event_type"] == "new_approval"),
        sum(1 for e in events if e["event_type"] == "phase3_complete"),
    )
    if not events:
        log.info("no notable events for LinkedIn posts — skipping LLM call")
        return []

    events = events[:max_posts]

    user_prompt = f"""\
Write one LinkedIn post for each of these FDA pipeline events:

{json.dumps(events, indent=2)}

Return ONLY valid JSON array: [{{"event_type": "...", "drug": "...", "text": "...", "hashtags": ["..."]}}]"""

    raw = _call(client, _model(cfg), _LINKEDIN_SYSTEM, user_prompt)
    if not raw:
        return None

    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        log.warning("failed to parse LinkedIn LLM response: %s", e)
        return None
