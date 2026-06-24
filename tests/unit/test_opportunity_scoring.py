"""Unit tests for opportunity scoring service (evidence-grounded scoring).

The score comes from Silver tag strengths (signal / theme / risk occurrences in
the retrieved chunks). Confidence comes from Gold trend coverage: depth
(coverage_count, saturating at depth_target) and recency (share of the recency
window's weeks with coverage).
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
              coverage=0, recent=0, window=4):
        return service.score_opportunity(
            risks=risks or [],
            signal_strength=signal,
            theme_strength=theme,
            risk_strength=risk,
            coverage_count=coverage,
            recent_weeks_covered=recent,
            recency_window=window,
        )

    # === Output shape ===

    def test_returns_dict_with_required_keys(self, service):
        result = self._call(service, signal=5, risk=4, coverage=10, recent=2)
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

    # === Confidence derives from Gold coverage depth + recency ===

    def test_full_depth_and_recency_is_max_confidence(self, service):
        result = self._call(service, coverage=service._depth_target, recent=4, window=4)
        assert result["confidence_level"] == 1.0

    def test_no_coverage_is_zero_confidence(self, service):
        result = self._call(service, coverage=0, recent=0, window=4)
        assert result["confidence_level"] == 0.0

    def test_half_depth_half_recency(self, service):
        # depth 0.5 * 0.6 + recency 0.5 * 0.4 = 0.5
        result = self._call(service, coverage=service._depth_target // 2, recent=2, window=4)
        assert result["confidence_level"] == 0.5

    def test_depth_saturates_at_target(self, service):
        result = self._call(service, coverage=service._depth_target * 10, recent=4, window=4)
        assert result["confidence_level"] == 1.0

    def test_depth_only_no_recency(self, service):
        # depth 1.0 * 0.6 + recency 0 = 0.6
        result = self._call(service, coverage=service._depth_target, recent=0, window=4)
        assert result["confidence_level"] == 0.6

    def test_recency_only_no_depth(self, service):
        # depth 0 + recency 1.0 * 0.4 = 0.4
        result = self._call(service, coverage=0, recent=4, window=4)
        assert result["confidence_level"] == 0.4

    def test_zero_window_is_no_recency(self, service):
        # window 0 -> recency factor 0, even with depth
        result = self._call(service, coverage=service._depth_target, recent=0, window=0)
        assert result["confidence_level"] == 0.6

    def test_confidence_within_unit_interval(self, service):
        for coverage, recent, window in ((0, 0, 4), (10, 1, 4), (9999, 4, 4)):
            c = self._call(service, coverage=coverage, recent=recent, window=window)["confidence_level"]
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
