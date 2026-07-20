"""Extracts an explicit date-range signal from a free-text retrieval query.
"""

import calendar
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional, Tuple

import dateparser
from dateparser.date import DateDataParser

logger = logging.getLogger(__name__)

_DATEPARSER_SETTINGS = {"PREFER_DATES_FROM": "past", "RETURN_AS_TIMEZONE_AWARE": True}
# get_date_data() exposes the parsed period (year/month/day).
_DATE_DATA_PARSER = DateDataParser(languages=["en"], settings=_DATEPARSER_SETTINGS)
_DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_YEAR_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:-|–|to|through|and)\s*((?:19|20)\d{2})\b"
)
# Splits a range span into its two bound texts on the connective
# ("2023-2024", "March 2022 to June 2024", "between 2022 and 2024"). The word
# forms are \b-anchored so they don't split inside a token; maxsplit keeps it
# to the first connective.
_RANGE_SPLIT_RE = re.compile(r"\s*(?:-|–|\bto\b|\bthrough\b|\band\b)\s*")
# Shorthand two-digit year range ("2023-24", fiscal-year style)
_SHORT_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*(?:-|–)\s*(\d{2})\b")


def _calendar_year_range(
    y0: int, y1: int, now: datetime
) -> Tuple[datetime, datetime]:
    """Full-calendar-year bounds: Jan 1 of y0 to Dec 31 of y1.

    Clamped to `now` so a range running past today never claims the future.
    """
    date_from = datetime(y0, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(y1, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    return date_from, min(date_to, now)


def _snap_bound(text: str, end: bool) -> Optional[datetime]:
    """Resolve one end of a range to the start or end of the date it names.

    Rounds off to however precise the bound is: a year ("2022") to Jan 1 /
    Dec 31, a month ("March 2022") to the first / last day of that month, a full
    day ("March 15 2022") to that day's 00:00 / 23:59:59. `end` picks the start
    vs the end. Returns None if the text has no parseable date, so the caller can
    fall back to the year-only path.
    """
    # Drop a leading connective the split left attached (e.g. the "between" in
    # "between 2022") so the parser sees only the date.
    cleaned = re.sub(r"^\s*(?:between|from)\s+", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = _DATE_DATA_PARSER.get_date_data(cleaned)
    except Exception:
        # Any parser hiccup degrades to "unresolved" so the caller falls back
        # rather than raising into the retrieval path.
        return None
    parsed = data["date_obj"]
    if not parsed:
        return None

    year, month, day = parsed.year, parsed.month, parsed.day
    precision = data["period"]  # "year" | "month" | "day"
    if precision == "year":
        month, day = (12, 31) if end else (1, 1)
    elif precision == "month":
        day = calendar.monthrange(year, month)[1] if end else 1
    hms = (23, 59, 59) if end else (0, 0, 0)
    return datetime(year, month, day, *hms, tzinfo=timezone.utc)


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

    # Relative window: "(the) last/past N days|weeks|months|years", or a bare
    # "N months". Scan for an adjacent <number> <unit> pair so leading filler
    # ("the", "over") does not defeat the match. Resolved with plain day-math
    # off `now` rather than dateparser, which resolves relative units off the
    # real current instant even when a caller passes a fixed `now` (tests).
    tokens = span.split()
    for i in range(len(tokens) - 1):
        num, unit = tokens[i], tokens[i + 1].rstrip("s")
        if num.isdigit() and unit in _DAYS_PER_UNIT:
            try:
                return now - timedelta(days=int(num) * _DAYS_PER_UNIT[unit]), now
            except OverflowError:
                # An absurd relative window ("last 99999999999 years") overflows
                # timedelta / date math. Degrade to "no intent" so retrieval
                # falls back to the default trailing window rather than crashing.
                logger.warning(
                    f"date-intent: relative window '{span}' overflowed; "
                    "falling back to default window"
                )
                return None

    # Shorthand two-digit year range ("2023-24"),century rolls over
    # when the pair is smaller than the leading year's own last two digits
    # (e.g. "1999-00" -> 1999-2000).
    short_match = _SHORT_YEAR_RANGE_RE.search(span)
    if short_match:
        y0 = int(short_match.group(1))
        y1 = (y0 // 100) * 100 + int(short_match.group(2))
        if y1 < y0:
            y1 += 100
        return _calendar_year_range(y0, y1, now)

    # Explicit range within the entity span ("2023-2024", "between 2022 and
    # 2024", "March 2022 to June 2024"). Split on the connective and resolve
    # each bound to its own precision (year, month, or day), then order
    # low -> high.
    parts = _RANGE_SPLIT_RE.split(span, maxsplit=1)
    if len(parts) == 2:
        start = _snap_bound(parts[0], end=False)
        end = _snap_bound(parts[1], end=True)
        if start and end:
            lo, hi = sorted((start, end))
            return lo, min(hi, now)

    # Fallback: a two-year span the split/snap above did not resolve (e.g. a
    # separator not in _RANGE_SPLIT_RE) still gets a year-granular range.
    years = _YEAR_RE.findall(span)
    if len(years) >= 2:
        ys = sorted(int(y) for y in years)
        return _calendar_year_range(ys[0], ys[-1], now)

    # Whole-query fallback: spaCy hands back only the first bound of some ranges
    # as the entity ("from 2022 to 2024" -> "2022"), so look for a connective
    # year range across the full query. The `\s*` adjacency keeps it from firing
    # on unrelated pairs ("2022 winners to 2024 losers").
    range_match = _YEAR_RANGE_RE.search(query)
    if range_match:
        ys = sorted(int(y) for y in range_match.groups())
        return _calendar_year_range(ys[0], ys[-1], now)

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
        return _calendar_year_range(y, y, now)

    return parsed, parsed