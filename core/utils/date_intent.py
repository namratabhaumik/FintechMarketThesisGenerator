"""Extracts an explicit date-range signal from a free-text retrieval query.
"""

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional, Tuple

import dateparser

logger = logging.getLogger(__name__)

_DATEPARSER_SETTINGS = {"PREFER_DATES_FROM": "past", "RETURN_AS_TIMEZONE_AWARE": True}
_RELATIVE_UNITS = ("day", "week", "month", "year")


@lru_cache(maxsize=1)
def _nlp():
    """Load spaCy once per process, NER-only (no tagger/parser/lemmatizer).

    Returns None (cached) if spaCy or its model can't be loaded - date-intent
    is an enhancement over the default trailing window, so a missing model must
    degrade to "no intent detected" rather than break the retrieval path. The
    lru_cache means a failed load is attempted once, not on every query.
    """
    try:
        import spacy

        return spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
        )
    except Exception as e:
        logger.warning(f"spaCy unavailable; date-intent parsing disabled: {e}")
        return None


def parse_date_intent(
    query: str, now: Optional[datetime] = None
) -> Optional[Tuple[Optional[datetime], Optional[datetime]]]:
    """Return (date_from, date_to) if the query names an explicit date range.

    Returns None if spaCy finds no DATE/TIME entity, so callers fall back to
    their default (e.g. the configured trailing window).
    """
    now = now or datetime.now(timezone.utc)

    nlp = _nlp()
    if nlp is None:
        return None

    try:
        doc = nlp(query)
        date_ents = [e.text for e in doc.ents if e.label_ in ("DATE", "TIME")]
    except Exception as e:
        # A NER hiccup must not sink retrieval: fall back to the caller's
        # default window instead of raising. (dateparser.parse below returns
        # None on failure and is None-checked, so it needs no guard here.)
        logger.warning(f"date-intent parsing failed for query '{query}': {e}")
        return None

    if not date_ents:
        return None

    span = date_ents[0].lower()

    # "last/past N days|weeks|months|years": resolve with plain day-math
    # rather than dateparser, which resolves relative units off the current
    # instant, not off `now` if a caller ever passes a fixed one (tests).
    words = span.replace("last", "").replace("past", "").split()
    if len(words) == 2 and words[0].isdigit() and words[1].rstrip("s") in _RELATIVE_UNITS:
        amount, unit = int(words[0]), words[1].rstrip("s")
        days_per_unit = {"day": 1, "week": 7, "month": 30, "year": 365}
        return now - timedelta(days=amount * days_per_unit[unit]), now

    parsed = dateparser.parse(date_ents[0], settings=_DATEPARSER_SETTINGS)
    if not parsed:
        return None
    parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    lower_query = query.lower()
    idx = lower_query.find(span)
    preceding = lower_query[max(0, idx - 12):idx]

    if any(w in preceding for w in ("before", "until", "up to")):
        return None, parsed
    if any(w in preceding for w in ("since", "after", "from")):
        return parsed, now

    # A bare date/year mention with no directional cue: treat it as "that
    # calendar year" rather than a single instant, when the entity is just a
    # year (dateparser otherwise snaps day/month to today's, e.g. "2024" ->
    # 2024-07-13 instead of the whole year).
    if span.strip().isdigit() and len(span.strip()) == 4:
        y = int(span.strip())
        date_from = datetime(y, 1, 1, tzinfo=timezone.utc)
        date_to = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return date_from, min(date_to, now)

    return parsed, parsed