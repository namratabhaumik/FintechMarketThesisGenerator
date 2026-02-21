"""Unit tests for KeywordCountScoringStrategy."""

import pytest
from core.implementations.keyword_scoring_strategy import KeywordCountScoringStrategy


@pytest.fixture
def strategy():
    return KeywordCountScoringStrategy()


class TestKeywordCountScoringStrategy:

    # === Return type ===

    def test_returns_dict(self, strategy):
        result = strategy.score("some text", {"Cat": ["keyword"]})
        assert isinstance(result, dict)

    def test_keys_match_category_map(self, strategy):
        category_map = {"Cat A": ["x"], "Cat B": ["y"]}
        result = strategy.score("x y z", category_map)
        assert set(result.keys()) == {"Cat A", "Cat B"}

    # === Counting ===

    def test_zero_for_no_match(self, strategy):
        result = strategy.score("unrelated text", {"Cat": ["payment", "banking"]})
        assert result["Cat"] == 0

    def test_one_for_single_keyword_match(self, strategy):
        result = strategy.score("the payment was processed", {"Cat": ["payment"]})
        assert result["Cat"] == 1

    def test_counts_each_keyword_once_regardless_of_occurrences(self, strategy):
        """'payment' appearing 5× still counts as 1 (keyword presence, not frequency)."""
        result = strategy.score(
            "payment payment payment payment payment",
            {"Cat": ["payment"]}
        )
        assert result["Cat"] == 1

    def test_counts_multiple_distinct_keywords(self, strategy):
        result = strategy.score(
            "payment and banking are both fintech topics",
            {"Cat": ["payment", "banking", "fintech"]}
        )
        assert result["Cat"] == 3

    def test_partial_keyword_match_counts(self, strategy):
        """Substring matching: 'pay' matches inside 'payment'."""
        result = strategy.score("payment processing", {"Cat": ["pay"]})
        assert result["Cat"] == 1

    # === Multiple categories ===

    def test_independent_scoring_per_category(self, strategy):
        category_map = {
            "Payments": ["payment", "transfer"],
            "Crypto":   ["blockchain", "crypto"],
        }
        text = "payment systems and blockchain technology are growing"
        result = strategy.score(text, category_map)
        assert result["Payments"] == 1
        assert result["Crypto"] == 1

    def test_zero_score_category_included_in_result(self, strategy):
        """Categories with no matches still appear in result with score 0."""
        result = strategy.score("unrelated text", {"Cat A": ["payment"], "Cat B": ["banking"]})
        assert "Cat A" in result
        assert "Cat B" in result
        assert result["Cat A"] == 0
        assert result["Cat B"] == 0

    # === Edge cases ===

    def test_empty_text_all_zeros(self, strategy):
        result = strategy.score("", {"Cat": ["payment", "banking"]})
        assert result["Cat"] == 0

    def test_empty_category_map_returns_empty_dict(self, strategy):
        result = strategy.score("payment banking fintech", {})
        assert result == {}

    def test_empty_keyword_list_returns_zero(self, strategy):
        result = strategy.score("payment banking", {"Cat": []})
        assert result["Cat"] == 0
