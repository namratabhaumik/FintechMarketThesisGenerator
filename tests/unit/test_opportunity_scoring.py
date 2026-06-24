"""Unit tests for opportunity scoring service (evidence-grounded scoring).

The score is computed from Silver tag strengths (signal / theme / risk
occurrences across the retrieved chunks), not from any prose. Confidence and key
risks are a separate, count-based assessment exposed via assess_confidence (used
on the refinement path, where the score is frozen).
"""

import pytest
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService


class TestOpportunityScoringService:
    """Tests for OpportunityScoringService evidence-grounded scoring."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return OpportunityScoringService()

    # === Output shape ===

    def test_score_opportunity_returns_dict_with_required_keys(self, service):
        """Test that scoring returns dict with required keys."""
        result = service.score_opportunity(
            risks=["Risk1"],
            investment_signals=["Signal1"],
            sources=["http://source1.com"],
            signal_strength=5,
            theme_strength=2,
            risk_strength=4,
        )
        assert isinstance(result, dict)
        for key in ("score", "confidence_level", "recommendation", "key_risks"):
            assert key in result

    # === Score derives from tag strengths ===

    def _score(self, service, signal, theme, risk):
        return service.score_opportunity(
            risks=[], investment_signals=[], sources=[],
            signal_strength=signal, theme_strength=theme, risk_strength=risk,
        )["score"]

    def test_no_strength_returns_base_score(self, service):
        """No evidence -> neutral base score of 2.5."""
        assert self._score(service, 0, 0, 0) == 2.5

    def test_signal_rich_scores_higher_than_risk_heavy(self, service):
        """Signal-dominant evidence outscores risk-dominant evidence."""
        high = self._score(service, 20, 0, 0)
        low = self._score(service, 0, 0, 20)
        assert high > low

    def test_signal_rich_is_pursue(self, service):
        """Strong signal strength -> high score -> Pursue."""
        result = service.score_opportunity(
            risks=[], investment_signals=[], sources=[],
            signal_strength=20, theme_strength=0, risk_strength=0,
        )
        assert result["score"] >= 3.75
        assert result["recommendation"] == "Pursue"

    def test_risk_heavy_is_skip(self, service):
        """Dominant risk strength -> low score -> Skip."""
        result = service.score_opportunity(
            risks=[], investment_signals=[], sources=[],
            signal_strength=0, theme_strength=0, risk_strength=20,
        )
        assert result["score"] < 2.5
        assert result["recommendation"] == "Skip"

    def test_balanced_is_investigate(self, service):
        """Mixed strengths -> mid score -> Investigate."""
        result = service.score_opportunity(
            risks=[], investment_signals=[], sources=[],
            signal_strength=5, theme_strength=0, risk_strength=4,
        )
        assert 2.5 <= result["score"] < 3.75
        assert result["recommendation"] == "Investigate"

    def test_more_signal_strength_raises_score(self, service):
        """The evidence lever: more signal strength -> higher score."""
        base = self._score(service, 5, 0, 4)
        boosted = self._score(service, 15, 0, 4)
        assert boosted > base

    def test_more_risk_strength_lowers_score(self, service):
        """More risk strength -> lower score."""
        base = self._score(service, 5, 0, 4)
        lowered = self._score(service, 5, 0, 14)
        assert lowered < base

    def test_themes_count_half_weight_toward_positive(self, service):
        """Theme strength lifts the score, but less than the same signal strength."""
        signal_only = self._score(service, 8, 0, 0)
        theme_only = self._score(service, 0, 8, 0)
        assert signal_only > theme_only > 2.5

    def test_score_clamped_between_0_and_5(self, service):
        """Score stays within [0, 5] for extreme inputs."""
        for signal, theme, risk in ((1000, 1000, 0), (0, 0, 1000), (0, 0, 0), (5, 2, 4)):
            score = self._score(service, signal, theme, risk)
            assert 0.0 <= score <= 5.0

    # === Confidence (assess_confidence: data-quality heuristic on label/source counts) ===

    def test_confidence_with_5_sources_3_signals_3_risks(self, service):
        """Typical case: source 1.0, signal 1.0, risk floor 0.5 -> 0.9."""
        result = service.assess_confidence(
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"],
        )
        assert result["confidence_level"] == 0.9

    def test_confidence_with_2_signals(self, service):
        """source 1.0, signal 2/3=0.667, risk 0.5 -> ~0.77."""
        result = service.assess_confidence(
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2"],
            sources=["url1", "url2", "url3", "url4", "url5"],
        )
        assert result["confidence_level"] == 0.77

    def test_confidence_with_1_signal(self, service):
        """source 1.0, signal 1/3=0.333, risk 0.5 -> ~0.63."""
        result = service.assess_confidence(
            risks=["R1", "R2", "R3"],
            investment_signals=["S1"],
            sources=["url1", "url2", "url3", "url4", "url5"],
        )
        assert result["confidence_level"] == 0.63

    def test_confidence_low_source_coverage(self, service):
        """source 0.2, signal 1.0, risk 1.0 -> 0.68."""
        result = service.assess_confidence(
            risks=[],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1"],
        )
        assert result["confidence_level"] == 0.68

    def test_confidence_high_risk_count(self, service):
        """risk_factor floored at 0.5 -> confidence 0.9."""
        result = service.assess_confidence(
            risks=["R1", "R2", "R3", "R4"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"],
        )
        assert result["confidence_level"] == 0.9

    def test_confidence_minimum_is_50_percent(self, service):
        """risk_factor minimum is 50% -> confidence 0.5 * 0.2 = 0.1."""
        result = service.assess_confidence(
            risks=["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
            investment_signals=[],
            sources=[],
        )
        assert result["confidence_level"] == 0.1

    def test_score_opportunity_confidence_matches_assess_confidence(self, service):
        """First-generation confidence uses the same heuristic as the refine path."""
        kwargs = dict(
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"],
        )
        full = service.score_opportunity(
            signal_strength=5, theme_strength=2, risk_strength=4, **kwargs
        )
        assert full["confidence_level"] == service.assess_confidence(**kwargs)["confidence_level"]

    # === Key Risks ===

    def test_key_risks_returns_top_3(self, service):
        result = service.assess_confidence(
            risks=["Risk1", "Risk2", "Risk3", "Risk4", "Risk5"],
            investment_signals=[],
            sources=["url1"],
        )
        assert result["key_risks"] == ["Risk1", "Risk2", "Risk3"]

    def test_key_risks_fewer_than_3(self, service):
        result = service.assess_confidence(
            risks=["Risk1", "Risk2"], investment_signals=[], sources=["url1"],
        )
        assert result["key_risks"] == ["Risk1", "Risk2"]

    def test_key_risks_empty(self, service):
        result = service.assess_confidence(
            risks=[], investment_signals=[], sources=["url1"],
        )
        assert result["key_risks"] == []

    # === Recommendation Boundaries ===

    def test_recommendation_boundaries(self, service):
        """Recommendation thresholds: >=3.75 Pursue, >=2.5 Investigate, else Skip."""
        assert service._get_recommendation(3.75) == "Pursue"
        assert service._get_recommendation(3.74) == "Investigate"
        assert service._get_recommendation(2.5) == "Investigate"
        assert service._get_recommendation(2.49) == "Skip"
