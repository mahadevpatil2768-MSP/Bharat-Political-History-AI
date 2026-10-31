"""Scope policy (no predictions/endorsements) and time-sensitivity / freshness checks."""

from __future__ import annotations

import re
from datetime import date

_OUT_OF_SCOPE = [
    re.compile(p, re.I) for p in (
        r"\b(who|which (party|alliance|candidate|leader))\b.{0,40}\b(will|would|is going to|gonna)\s+win\b",
        r"\bpredict\w*\b.{0,60}\b(election|poll|vote|seats?|outcome|result)",
        r"\b(chances?|probability|odds|likelihood)\b.{0,40}\b(win|winning|victory|election|poll)",
        r"\b(who|whom|which party)\s+should\s+(i|we|people|one)\s+vote",
        r"\bshould\s+(i|we)\s+(vote|support)\b",
        r"\bwhich\s+(party|leader|candidate|ideology|government)\s+is\s+(better|the best|best|worse|the worst)\b",
        r"\b(endorse|recommend)\b.{0,30}\b(party|candidate|leader)\b",
    )
]

DECLINE_MESSAGE = (
    "This tool is a neutral historical research aid. It does not predict election outcomes, estimate "
    "probabilities of political events, or recommend or endorse parties, candidates, ideologies or "
    "voting choices. You can instead ask what the indexed sources document — for example past results, "
    "manifestos, legislation or constitutional provisions."
)

_TIME_SENSITIVE = re.compile(
    r"\b(current(ly)?|latest|today|now|nowadays|present(ly)?|recent(ly)?|incumbent|ongoing|upcoming|"
    r"this year|so far|as of now|at present|sitting)\b", re.I)


def is_out_of_scope(question: str) -> bool:
    return any(p.search(question) for p in _OUT_OF_SCOPE)


def is_time_sensitive(question: str, today: date | None = None) -> bool:
    today = today or date.today()
    if _TIME_SENSITIVE.search(question):
        return True
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    return any(y >= today.year - 1 for y in years)


def parse_partial_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        parts = [int(p) for p in s.split("-")]
        return date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
    except (ValueError, IndexError):
        return None


def freshness_notices(pub_dates: list[str | None], max_age_days: int, today: date | None = None) -> list[str]:
    """Notices for time-sensitive questions. No web search exists, so always disclose that."""
    today = today or date.today()
    notices = ["This question appears time-sensitive. Answers are limited to the indexed documents — "
               "no live web or news search is performed — so recent developments may be missing."]
    parsed = [d for d in (parse_partial_date(p) for p in pub_dates) if d]
    if not parsed:
        notices.append("None of the retrieved sources has a known publication date, so their currency "
                       "cannot be assessed.")
        return notices
    newest = max(parsed)
    age = (today - newest).days
    if age > max_age_days:
        notices.append(f"The newest retrieved source was published {newest.isoformat()} "
                       f"({age} days ago, older than the {max_age_days}-day freshness window). "
                       "Treat it as historical, not as a description of the present.")
    return notices
