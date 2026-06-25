"""Unit tests for opportunity scoring service (evidence-grounded scoring).

The score comes from Silver tag strengths (signal / theme / risk occurrences in
the retrieved chunks). Confidence comes from Gold trend coverage: the share of
the evidence window's weeks (covered_weeks / window_weeks) in which the corpus
covered the thesis's categories.
"""

import pytest
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService


class TestOpportunityScoringService:
    """Tests for OpportunityScoringService evidence-grounded scoring."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return OpportunityScoringService()

    def _call(self, service, *, signal=0, theme=0, risk=0, risks=None,
              covered=0, window=4):
        return service.score_opportunity(
            risks=risks or [],
            signal_strength=signal,
            theme_strength=theme,
            risk_strength=risk,
            covered_weeks=covered,
            window_weeks=window,
        )

    # === Output shape ===

    def test_returns_dict_with_required_keys(self, service):
        result = self._call(service, signal=5, risk=4, covered=2, window=4)
        assert isinstance(result, dict)
        for key in ("score", "confidence_level", "recommendation", "key_risks"):
            assert key in result

    # === Score derives from Silver tag strengths ===

    def _score(self, service, signal, theme, risk):
        return self._call(service, signal=signal, theme=theme, risk=risk)["score"]

    def test_no_strength_returns_base_score(self, service):
        """No evidence -> neutral base score of 2.5."""
        assert self._score(service, 0, 0, 0) == 2.5

    def test_signal_rich_scores_higher_than_risk_heavy(self, service):
        assert self._score(service, 20, 0, 0) > self._score(service, 0, 0, 20)

    def test_signal_rich_is_pursue(self, service):
        result = self._call(service, signal=20)
        assert result["score"] >= 3.75
        assert result["recommendation"] == "Pursue"

    def test_risk_heavy_is_skip(self, service):
        result = self._call(service, risk=20)
        assert result["score"] < 2.5
        assert result["recommendation"] == "Skip"

    def test_balanced_is_investigate(self, service):
        result = self._call(service, signal=5, risk=4)
        assert 2.5 <= result["score"] < 3.75
        assert result["recommendation"] == "Investigate"

    def test_more_signal_strength_raises_score(self, service):
        assert self._score(service, 15, 0, 4) > self._score(service, 5, 0, 4)

    def test_more_risk_strength_lowers_score(self, service):
        assert self._score(service, 5, 0, 14) < self._score(service, 5, 0, 4)

    def test_themes_count_half_weight_toward_positive(self, service):
        signal_only = self._score(service, 8, 0, 0)
        theme_only = self._score(service, 0, 8, 0)
        assert signal_only > theme_only > 2.5

    def test_score_clamped_between_0_and_5(self, service):
        for signal, theme, risk in ((1000, 1000, 0), (0, 0, 1000), (0, 0, 0), (5, 2, 4)):
            assert 0.0 <= self._score(service, signal, theme, risk) <= 5.0

    # === Confidence = covered_weeks / window_weeks (Gold coverage over the window) ===

    def test_all_weeks_covered_is_max_confidence(self, service):
        result = self._call(service, covered=4, window=4)
        assert result["confidence_level"] == 1.0

    def test_no_weeks_covered_is_zero_confidence(self, service):
        result = self._call(service, covered=0, window=4)
        assert result["confidence_level"] == 0.0

    def test_half_the_weeks_covered(self, service):
        result = self._call(service, covered=2, window=4)
        assert result["confidence_level"] == 0.5

    def test_fraction_over_long_window(self, service):
        # 39 of 52 weeks covered -> 0.75
        result = self._call(service, covered=39, window=52)
        assert result["confidence_level"] == 0.75

    def test_zero_window_is_zero_confidence(self, service):
        # No window (e.g. empty Gold) -> 0.0, never a divide-by-zero.
        result = self._call(service, covered=0, window=0)
        assert result["confidence_level"] == 0.0

    def test_covered_exceeding_window_clamps_to_one(self, service):
        result = self._call(service, covered=10, window=4)
        assert result["confidence_level"] == 1.0

    def test_confidence_within_unit_interval(self, service):
        for covered, window in ((0, 4), (1, 4), (52, 52), (10, 4), (0, 0)):
            c = self._call(service, covered=covered, window=window)["confidence_level"]
            assert 0.0 <= c <= 1.0

    # === Key Risks ===

    def test_key_risks_returns_top_3(self, service):
        result = self._call(service, risks=["R1", "R2", "R3", "R4", "R5"])
        assert result["key_risks"] == ["R1", "R2", "R3"]

    def test_key_risks_fewer_than_3(self, service):
        result = self._call(service, risks=["R1", "R2"])
        assert result["key_risks"] == ["R1", "R2"]

    def test_key_risks_empty(self, service):
        result = self._call(service, risks=[])
        assert result["key_risks"] == []

    # === Recommendation Boundaries ===

    def test_recommendation_boundaries(self, service):
        """Recommendation thresholds: >=3.75 Pursue, >=2.5 Investigate, else Skip."""
        assert service._get_recommendation(3.75) == "Pursue"
        assert service._get_recommendation(3.74) == "Investigate"
        assert service._get_recommendation(2.5) == "Investigate"
        assert service._get_recommendation(2.49) == "Skip"
