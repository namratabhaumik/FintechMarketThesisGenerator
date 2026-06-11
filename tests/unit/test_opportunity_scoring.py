"""Unit tests for opportunity scoring service (content-strength scoring)."""

import pytest
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService


# Thesis-prose snippets with known fintech keyword density (see calibration).
SIGNAL_RICH = (
    "Startups are scaling embedded finance and BaaS APIs with strong BNPL, "
    "real-time payment rails, AI-driven financial tools, robo-advisor wealthtech "
    "and crypto/DeFi for retail investors and gen z in emerging markets like "
    "India, Africa and LATAM."
)
RISK_HEAVY = (
    "Regulatory enforcement and compliance crackdowns, SEC bans, data breaches, "
    "fraud and phishing, intense competition from incumbents and big tech, credit "
    "risk, default and liquidity concerns amid recession, inflation and rising "
    "interest rates."
)
BALANCED = (
    "Digital payments and neobank adoption grow with embedded finance and API "
    "platforms, but regulatory compliance, competition and cybersecurity fraud "
    "risks remain headwinds for lending and BNPL."
)


class TestOpportunityScoringService:
    """Tests for OpportunityScoringService content-strength scoring."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return OpportunityScoringService()

    # === Output shape ===

    def test_score_opportunity_returns_dict_with_required_keys(self, service):
        """Test that scoring returns dict with required keys."""
        result = service.score_opportunity(
            key_themes=["Theme1"],
            risks=["Risk1"],
            investment_signals=["Signal1"],
            sources=["http://source1.com"],
            raw_text=BALANCED,
        )
        assert isinstance(result, dict)
        for key in ("score", "confidence_level", "recommendation", "key_risks"):
            assert key in result

    # === Score derives from content strength, not list lengths ===

    def test_empty_content_returns_base_score(self, service):
        """No content -> neutral base score of 2.5, regardless of label counts."""
        result = service.score_opportunity(
            key_themes=["T1", "T2", "T3"],
            risks=["R1"],
            investment_signals=["S1", "S2"],
            sources=["url1"],
            raw_text="",
        )
        assert result["score"] == 2.5

    def test_signal_rich_scores_higher_than_risk_heavy(self, service):
        """Signal-heavy prose outscores risk-heavy prose."""
        high = service.score_opportunity([], [], [], [], raw_text=SIGNAL_RICH)["score"]
        low = service.score_opportunity([], [], [], [], raw_text=RISK_HEAVY)["score"]
        assert high > low

    def test_signal_rich_is_pursue(self, service):
        """Strong signal language -> high score -> Pursue."""
        result = service.score_opportunity([], [], [], [], raw_text=SIGNAL_RICH)
        assert result["score"] >= 3.75
        assert result["recommendation"] == "Pursue"

    def test_risk_heavy_is_skip(self, service):
        """Dominant risk language -> low score -> Skip."""
        result = service.score_opportunity([], [], [], [], raw_text=RISK_HEAVY)
        assert result["score"] < 2.5
        assert result["recommendation"] == "Skip"

    def test_balanced_is_investigate(self, service):
        """Mixed language -> mid score -> Investigate."""
        result = service.score_opportunity([], [], [], [], raw_text=BALANCED)
        assert 2.5 <= result["score"] < 3.75
        assert result["recommendation"] == "Investigate"

    def test_adding_signal_language_raises_score(self, service):
        """The refinement lever: more signal language -> higher score."""
        base = service.score_opportunity([], [], [], [], raw_text=BALANCED)["score"]
        boosted = service.score_opportunity(
            [], [], [], [],
            raw_text=BALANCED + " embedded finance baas bnpl robo-advisor crypto defi "
                                "ai-driven real-time payment instant payment wealthtech",
        )["score"]
        assert boosted > base

    def test_adding_risk_language_lowers_score(self, service):
        """More risk language -> lower score."""
        base = service.score_opportunity([], [], [], [], raw_text=BALANCED)["score"]
        lowered = service.score_opportunity(
            [], [], [], [],
            raw_text=BALANCED + " regulatory enforcement ban breach hack fraud default "
                                "liquidity recession sanction outage downtime",
        )["score"]
        assert lowered < base

    def test_score_clamped_between_0_and_5(self, service):
        """Score stays within [0, 5] for extreme inputs."""
        for txt in (SIGNAL_RICH * 5, RISK_HEAVY * 5, "", BALANCED):
            score = service.score_opportunity([], [], [], [], raw_text=txt)["score"]
            assert 0.0 <= score <= 5.0

    # === Confidence Calculation Tests (data-quality heuristic on label/source counts) ===

    def test_confidence_with_5_sources_3_signals_3_risks(self, service):
        """Test confidence calculation with typical case."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source_factor=1.0, signal_factor=1.0, risk_factor=0.5
        # confidence = 0.4 + 0.4 + 0.1 = 0.9
        assert result["confidence_level"] == 0.9

    def test_confidence_with_2_signals(self, service):
        """Test confidence with 2 signals."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=["S1", "S2"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source 1.0, signal 2/3=0.667, risk 0.5 -> ~0.77
        assert result["confidence_level"] == 0.77

    def test_confidence_with_1_signal(self, service):
        """Test confidence with 1 signal (weak case)."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3"],
            investment_signals=["S1"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # source 1.0, signal 1/3=0.333, risk 0.5 -> ~0.63
        assert result["confidence_level"] == 0.63

    def test_confidence_low_source_coverage(self, service):
        """Test confidence with fewer sources."""
        result = service.score_opportunity(
            key_themes=[],
            risks=[],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1"]
        )
        # source 0.2, signal 1.0, risk 1.0 -> 0.68
        assert result["confidence_level"] == 0.68

    def test_confidence_high_risk_count(self, service):
        """Test confidence reduced by high risk count."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3", "R4"],
            investment_signals=["S1", "S2", "S3"],
            sources=["url1", "url2", "url3", "url4", "url5"]
        )
        # risk_factor floored at 0.5 -> confidence 0.9
        assert result["confidence_level"] == 0.9

    def test_confidence_minimum_is_50_percent(self, service):
        """Test that risk_factor minimum is 50%."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
            investment_signals=[],
            sources=[]
        )
        # confidence = 0.5 * 0.2 = 0.1
        assert result["confidence_level"] == 0.1

    # === Key Risks Tests ===

    def test_key_risks_returns_top_3(self, service):
        """Test that key_risks returns top 3 risks."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["Risk1", "Risk2", "Risk3", "Risk4", "Risk5"],
            investment_signals=[],
            sources=["url1"]
        )
        assert result["key_risks"] == ["Risk1", "Risk2", "Risk3"]

    def test_key_risks_fewer_than_3(self, service):
        """Test key_risks when fewer than 3 risks."""
        result = service.score_opportunity(
            key_themes=[],
            risks=["Risk1", "Risk2"],
            investment_signals=[],
            sources=["url1"]
        )
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

    # === Recommendation Boundary Tests ===

    def test_recommendation_boundaries(self, service):
        """Recommendation thresholds: >=3.75 Pursue, >=2.5 Investigate, else Skip."""
        assert service._get_recommendation(3.75) == "Pursue"
        assert service._get_recommendation(3.74) == "Investigate"
        assert service._get_recommendation(2.5) == "Investigate"
        assert service._get_recommendation(2.49) == "Skip"
