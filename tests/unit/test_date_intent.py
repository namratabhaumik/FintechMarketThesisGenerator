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
