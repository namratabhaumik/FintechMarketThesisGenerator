"""Unit tests for opportunity scoring service."""

import pytest
from core.services.opportunity_scoring_service import OpportunityScoringService


class TestOpportunityScoringService:
    """Tests for OpportunityScoringService rule-based scoring."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return OpportunityScoringService()

    # === Scoring Calculation Tests ===

    def test_score_opportunity_returns_dict_with_required_keys(self, service):
        """Test that scoring returns dict with required keys."""
        result = service.score_opportunity(
            key_themes=["Theme1"],
            risks=["Risk1"],
            investment_signals=["Signal1"],
            sources=["http://source1.com"]
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "confidence_level" in result
        assert "recommendation" in result
        assert "key_risks" in result

    def test_score_clamped_between_0_and_5(self, service):
        """Test that score is clamped between 0 and 5."""
        # High score case
        result_high = service.score_opportunity(
            key_themes=["T1", "T2", "T3"],
            risks=[],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3"]
        )
        assert 0 <= result_high["score"] <= 5.0

        # Low score case
        result_low = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3", "R4", "R5"],
            investment_signals=[],
            sources=[]
        )
        assert 0 <= result_low["score"] <= 5.0

    def test_base_score_no_signals(self, service):
        """Test base score with no signals or themes."""
        result = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=[],
            sources=["url1"]
        )
        # Base score = 2.5, no additions
        assert result["score"] == 2.5

    def test_scoring_formula_with_signals(self, service):
        """Test scoring includes signal weight."""
        result = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=["Signal1"],
            sources=["url1"]
        )
        # score = 2.5 (base) + 0.75 (1 signal × 0.75 weight) = 3.25 → rounds to 3.2
        assert result["score"] == 3.2

    def test_scoring_formula_with_themes(self, service):
        """Test scoring includes theme weight."""
        result = service.score_opportunity(
            key_themes=["Theme1"],
            risks=[],
            investment_signals=[],
            sources=["url1"]
        )
        # score = 2.5 (base) + 0.25 (1 theme × 0.25 weight) = 2.75 → rounds to 2.8
        assert result["score"] == 2.8

    def test_scoring_formula_with_risks(self, service):
        """Test scoring subtracts risk penalty."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["Risk1"],
            investment_signals=[],
            sources=["url1"]
        )
        # score = 2.5 (base) - 0.25 (1 risk × 0.25 penalty) = 2.25 → rounds to 2.2
        assert result["score"] == 2.2

    def test_scoring_respects_max_themes(self, service):
        """Test that scoring caps themes at max_themes."""
        result_3_themes = service.score_opportunity(
            key_themes=["T1", "T2", "T3"],
            risks=[],
            investment_signals=[],
            sources=["url1"]
        )
        result_5_themes = service.score_opportunity(
            key_themes=["T1", "T2", "T3", "T4", "T5"],
            risks=[],
            investment_signals=[],
            sources=["url1"]
        )
        # Both should have same score since max_themes=3
        assert result_3_themes["score"] == result_5_themes["score"]

    def test_scoring_respects_max_signals(self, service):
        """Test that scoring caps signals at max_signals."""
        result_3_signals = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1"]
        )
        result_5_signals = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=["S1", "S2", "S3", "S4", "S5"],
            sources=["url1"]
        )
        # Both should have same score since max_signals=3
        assert result_3_signals["score"] == result_5_signals["score"]

    def test_scoring_respects_max_risks(self, service):
        """Test that risk penalty caps at max_risks."""
        result_3_risks = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=[],
            sources=["url1"]
        )
        result_5_risks = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3", "R4", "R5"],
            investment_signals=[],
            sources=["url1"]
        )
        # Both should have same score since max_risks=3
        assert result_3_risks["score"] == result_5_risks["score"]

    # === Confidence Calculation Tests ===

    def test_confidence_with_5_sources_3_signals_3_risks(self, service):
        """Test confidence calculation with typical case."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source_factor = min(5/5, 1.0) = 1.0
        # signal_factor = min(3/3, 1.0) = 1.0
        # risk_factor = max(0.5, 1.0 - 3/4) = max(0.5, 0.25) = 0.5
        # confidence = (1.0 × 0.4) + (1.0 × 0.4) + (0.5 × 0.2) = 0.4 + 0.4 + 0.1 = 0.9
        assert result["confidence_level"] == 0.9

    def test_confidence_with_2_signals(self, service):
        """Test confidence with 2 signals (Investigate case)."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source_factor = 1.0, signal_factor = 2/3 = 0.667, risk_factor = 0.5
        # confidence = (1.0 × 0.4) + (0.667 × 0.4) + (0.5 × 0.2) ≈ 0.77
        assert result["confidence_level"] == 0.77

    def test_confidence_with_1_signal(self, service):
        """Test confidence with 1 signal (weak case)."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=["S1"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source_factor = 1.0, signal_factor = 1/3 = 0.333, risk_factor = 0.5
        # confidence = (1.0 × 0.4) + (0.333 × 0.4) + (0.5 × 0.2) ≈ 0.63
        assert result["confidence_level"] == 0.63

    def test_confidence_low_source_coverage(self, service):
        """Test confidence with fewer sources."""
        result = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1"]
        )
        # source_factor = 1/5 = 0.2, signal_factor = 1.0, risk_factor = 1.0
        # confidence = (0.2 × 0.4) + (1.0 × 0.4) + (1.0 × 0.2) = 0.08 + 0.4 + 0.2 = 0.68
        assert result["confidence_level"] == 0.68

    def test_confidence_high_risk_count(self, service):
        """Test confidence reduced by high risk count."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3", "R4"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source_factor = 1.0, signal_factor = 1.0
        # risk_factor = max(0.5, 1.0 - 4/4) = max(0.5, 0.0) = 0.5
        # confidence = (1.0 × 0.4) + (1.0 × 0.4) + (0.5 × 0.2) = 0.9
        assert result["confidence_level"] == 0.9

    def test_confidence_minimum_is_50_percent(self, service):
        """Test that risk_factor minimum is 50%."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
            investment_signals=[],
            sources=[]
        )
        # risk_factor should be at least 0.5
        # source_factor = 0, signal_factor = 0
        # confidence = (0 × 0.4) + (0 × 0.4) + (0.5 × 0.2) = 0.1
        assert result["confidence_level"] == 0.1

    # === Recommendation Tests ===

    def test_recommendation_pursue_high_score(self, service):
        """Test 'Pursue' recommendation for high score."""
        result = service.score_opportunity(
            key_themes=["T1"],
            risks=[],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1"]
        )
        # score = 2.5 + 0.25 + 2.25 = 5.0 (clamped)
        assert result["score"] >= 3.75
        assert result["recommendation"] == "Pursue"

    def test_recommendation_investigate_mid_score(self, service):
        """Test 'Investigate' recommendation for mid-range score."""
        result = service.score_opportunity(
            key_themes=["T1"],
            risks=["R1"],
            investment_signals=["S1", "S2"],
            sources=["url1"]
        )
        # score = 2.5 + 0.25 + 1.5 - 0.25 = 4.0... but let's calculate exactly
        # Actually: 2.5 + 0.25 (1 theme) + 1.5 (2 signals) - 0.25 (1 risk) = 4.0
        # That's Pursue. Let me adjust
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1"],
            investment_signals=["S1"],
            sources=["url1"]
        )
        # score = 2.5 + 0.75 - 0.25 = 3.0
        assert 2.5 <= result["score"] < 3.75
        assert result["recommendation"] == "Investigate"

    def test_recommendation_skip_low_score(self, service):
        """Test 'Skip' recommendation for low score."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2"],
            investment_signals=[],
            sources=["url1"]
        )
        # score = 2.5 - 0.5 = 2.0
        assert result["score"] < 2.5
        assert result["recommendation"] == "Skip"

    # === Key Risks Tests ===

    def test_key_risks_returns_top_3(self, service):
        """Test that key_risks returns top 3 risks."""
        risks = ["Risk1", "Risk2", "Risk3", "Risk4", "Risk5"]
        result = service.score_opportunity(
            key_themes=[],
            risks=risks,
            investment_signals=[],
            sources=["url1"]
        )
        assert len(result["key_risks"]) == 3
        assert result["key_risks"] == ["Risk1", "Risk2", "Risk3"]

    def test_key_risks_fewer_than_3(self, service):
        """Test key_risks when fewer than 3 risks."""
        risks = ["Risk1", "Risk2"]
        result = service.score_opportunity(
            key_themes=[],
            risks=risks,
            investment_signals=[],
            sources=["url1"]
        )
        assert len(result["key_risks"]) == 2
        assert result["key_risks"] == ["Risk1", "Risk2"]

    def test_key_risks_empty(self, service):
        """Test key_risks when no risks."""
        result = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=[],
            sources=["url1"]
        )
        assert result["key_risks"] == []

    # === Real-World Scenario Tests ===

    def test_strong_opportunity_high_confidence(self, service):
        """Test strong opportunity case (Pursue with high confidence)."""
        result = service.score_opportunity(
            key_themes=["AI-Powered Automation", "Digital Payments"],
            risks=["Market Adoption Risk"],
            investment_signals=["AI-Driven Financial Tools", "Payment Infrastructure", "Embedded Finance Opportunity"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # score = 2.5 + 0.25*2 + 0.75*3 - 0.25*1 = 2.5 + 0.5 + 2.25 - 0.25 = 5.0
        # confidence: source_factor=1.0, signal_factor=1.0, risk_factor=0.75
        # confidence = (1.0*0.4) + (1.0*0.4) + (0.75*0.2) = 0.95
        assert result["score"] == 5.0
        assert result["recommendation"] == "Pursue"
        assert result["confidence_level"] == 0.95

    def test_moderate_opportunity_moderate_confidence(self, service):
        """Test moderate opportunity case (Investigate)."""
        result = service.score_opportunity(
            key_themes=["Neobanking"],
            risks=["Regulatory Risk", "Cybersecurity Risk"],
            investment_signals=["Consumer Fintech Adoption"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # score = 2.5 + 0.25*1 + 0.75*1 - 0.25*2 = 2.5 + 0.25 + 0.75 - 0.5 = 3.0
        assert result["recommendation"] == "Investigate"
        assert 0.5 <= result["confidence_level"] <= 0.9

    def test_weak_opportunity_low_confidence(self, service):
        """Test weak opportunity case (Skip)."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["Regulatory Risk", "Cybersecurity Risk", "Market Adoption Risk"],
            investment_signals=[],
            sources=["url1"]
        )
        # score = 2.5 - 0.25*3 = 2.5 - 0.75 = 1.75
        assert result["score"] < 2.5
        assert result["recommendation"] == "Skip"

    # === Recommendation Boundary Tests ===

    def test_score_exactly_3_75_is_pursue(self, service):
        """Test that score of exactly 3.75 triggers Pursue."""
        # Create scenario with exactly 3.75
        # 2.5 + 0.25*3 + 0.75*1 - 0.25*0 = 2.5 + 0.75 + 0.75 = 4.0 (too high)
        # Let's use: 2.5 + 0.75*1 + 0.25*1 - 0.25*0.4 = 2.5 + 0.75 + 0.25 - 0.1 = 3.4 (too low)
        # Actually, let's just check the method directly
        assert service._get_recommendation(3.75) == "Pursue"
        assert service._get_recommendation(3.74) == "Investigate"

    def test_score_exactly_2_5_is_investigate(self, service):
        """Test that score of exactly 2.5 triggers Investigate."""
        assert service._get_recommendation(2.5) == "Investigate"
        assert service._get_recommendation(2.49) == "Skip"
