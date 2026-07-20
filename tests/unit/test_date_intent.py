"""Unit tests for date_intent.parse_date_intent."""

from datetime import datetime, timezone

from core.utils.date_intent import parse_date_intent

NOW = datetime(2026, 7, 13, tzinfo=timezone.utc)


class TestParseDateIntent:
    """No date intent: an ordinary semantic query must not get date-filtered,
    even when it contains domain phrases that a naive date scan misreads."""

    def test_plain_query_returns_none(self):
        assert parse_date_intent("crypto adoption in Asia", now=NOW) is None

    def test_buy_now_pay_later_is_not_a_date(self):
        assert parse_date_intent("buy now pay later regulations", now=NOW) is None

    def test_may_as_a_verb_is_not_a_date(self):
        assert parse_date_intent("digital banking may grow", now=NOW) is None


class TestExplicitYear:
    """A bare year names that whole calendar year, not a single instant."""

    def test_bare_year_gives_full_year_range(self):
        date_from, date_to = parse_date_intent("fintech trends in 2024", now=NOW)
        assert date_from == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_current_year_end_is_clamped_to_now(self):
        _, date_to = parse_date_intent("fintech trends in 2026", now=NOW)
        assert date_to == NOW


class TestDirectionalPhrases:
    def test_since_gives_open_ended_range_to_now(self):
        date_from, date_to = parse_date_intent(
            "stablecoin adoption since March 2024", now=NOW
        )
        assert date_from is not None
        assert date_from.year == 2024 and date_from.month == 3
        assert date_to == NOW

    def test_before_gives_open_start_range(self):
        date_from, date_to = parse_date_intent("crypto in Asia before 2023", now=NOW)
        assert date_from is None
        assert date_to is not None and date_to.year == 2023


class TestRelativePhrases:
    def test_last_n_months_anchors_to_now(self):
        date_from, date_to = parse_date_intent("fintech news last 6 months", now=NOW)
        assert date_to == NOW
        assert (NOW - date_from).days == 6 * 30

    def test_leading_filler_does_not_defeat_relative_window(self):
        # "the"/"over the" ahead of the count must still resolve (regression:
        # the old two-token check only matched a bare "N months").
        date_from, date_to = parse_date_intent(
            "neobank margins over the last 6 months", now=NOW
        )
        assert date_to == NOW
        assert (NOW - date_from).days == 6 * 30

    def test_absurd_relative_window_degrades_to_none(self):
        # An overflowing count must fall back to the default window, not crash.
        assert parse_date_intent("trends in the last 99999999999 years", now=NOW) is None


class TestYearRanges:
    """Two-year spans resolve to an earliest -> latest full-year range."""

    def test_hyphen_range(self):
        date_from, date_to = parse_date_intent(
            "BNPL outlook after the 2023-2024 correction", now=NOW
        )
        assert date_from == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_between_x_and_y(self):
        date_from, date_to = parse_date_intent(
            "fintech IPOs between 2022 and 2024", now=NOW
        )
        assert date_from == datetime(2022, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_from_x_to_y_whole_query_fallback(self):
        # spaCy hands back only the first bound ("2022") as the entity here, so
        # this exercises the whole-query connective fallback.
        date_from, date_to = parse_date_intent("crypto from 2022 to 2024", now=NOW)
        assert date_from == datetime(2022, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_unrelated_year_pair_is_not_a_range(self):
        # Non-adjacent years joined by prose must not be read as a range.
        assert parse_date_intent("2022 winners to 2024 losers", now=NOW) is None


class TestShorthandYearRanges:
    """Fiscal-year-style two-digit shorthand ("2023-24") expands using the
    leading year's century, matching the equivalent full "2023-2024" form -
    rather than handing the bare "24" to the generic parser, which reads it
    as day-24-of-the-current-month."""

    def test_shorthand_matches_full_year_range(self):
        date_from, date_to = parse_date_intent(
            "fintech trends between 2023-24", now=NOW
        )
        assert date_from == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_century_rolls_over_past_year_end(self):
        date_from, date_to = parse_date_intent(
            "outlook for the 1999-00 season", now=NOW
        )
        assert date_from == datetime(1999, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2000, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_full_four_digit_range_is_unaffected(self):
        # Guards against the shorthand pattern misfiring on "2023-2024": \d{2}
        # can't reach a word boundary two digits into a four-digit year.
        date_from, date_to = parse_date_intent(
            "fintech trends between 2023-2024", now=NOW
        )
        assert date_from == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


class TestMonthGranularityRanges:
    """A range whose bounds name months resolves at month grain, not full year."""

    def test_month_to_month(self):
        date_from, date_to = parse_date_intent(
            "fintech from March 2022 to June 2024", now=NOW
        )
        assert date_from == datetime(2022, 3, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

    def test_mixed_year_start_month_end(self):
        # Each bound is resolved at its own grain: year-start, month-end.
        date_from, date_to = parse_date_intent(
            "lending from 2022 to June 2024", now=NOW
        )
        assert date_from == datetime(2022, 1, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

    def test_month_end_respects_leap_february(self):
        date_from, date_to = parse_date_intent(
            "trends from Feb 2024 to February 2024", now=NOW
        )
        assert date_from == datetime(2024, 2, 1, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 2, 29, 23, 59, 59, tzinfo=timezone.utc)

    def test_explicit_days_snap_to_that_day(self):
        # A bound naming a day resolves to that exact day, not the whole month.
        date_from, date_to = parse_date_intent(
            "fintech from March 15 2022 to June 3 2024", now=NOW
        )
        assert date_from == datetime(2022, 3, 15, tzinfo=timezone.utc)
        assert date_to == datetime(2024, 6, 3, 23, 59, 59, tzinfo=timezone.utc)
