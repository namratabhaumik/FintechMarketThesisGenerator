"""Unit tests for thesis structuring service."""

import pytest
from core.services.thesis_structuring_service import ThesisStructuringService
from core.implementations.keyword_scoring_strategy import KeywordCountScoringStrategy
from core.services.category_mappings import ThemeMappings, RiskMappings, SignalMappings


class TestThesisStructuringService:
    """Tests for ThesisStructuringService keyword-to-category mapping."""

    @pytest.fixture
    def service(self):
        """Create service instance with real scoring strategy."""
        return ThesisStructuringService(KeywordCountScoringStrategy())

    # === Basic Functionality Tests ===

    def test_structure_thesis_returns_dict_with_required_keys(self, service):
        """Test that structuring returns dict with required keys."""
        result = service.structure_thesis("Some summary text")

        assert isinstance(result, dict)
        assert "key_themes" in result
        assert "risks" in result
        assert "investment_signals" in result

    def test_structure_thesis_returns_lists(self, service):
        """Test that all values are lists."""
        result = service.structure_thesis("Some summary text")

        assert isinstance(result["key_themes"], list)
        assert isinstance(result["risks"], list)
        assert isinstance(result["investment_signals"], list)

    def test_get_structurer_name(self, service):
        """Test that structurer name is returned."""
        assert service.get_structurer_name() == "KeywordMappingStructurer"

    # === Single Category Match Tests ===

    def test_match_neobanking_theme(self, service):
        """Test matching Neobanking theme with 'digital bank' keyword."""
        summary = "This neobank offers digital banking solutions"
        result = service.structure_thesis(summary)

        assert "Neobanking" in result["key_themes"]

    def test_match_digital_payments_theme(self, service):
        """Test matching Digital Payments theme."""
        summary = "The new payment platform enables peer-to-peer transfers"
        result = service.structure_thesis(summary)

        assert "Digital Payments" in result["key_themes"]

    def test_match_blockchain_theme(self, service):
        """Test matching Blockchain & Web3 theme."""
        summary = "We built a blockchain-based wallet for crypto assets"
        result = service.structure_thesis(summary)

        assert "Blockchain & Web3" in result["key_themes"]

    def test_match_regulatory_risk(self, service):
        """Test matching Regulatory Risk."""
        summary = "The SEC enforcement action shows regulatory challenges"
        result = service.structure_thesis(summary)

        assert "Regulatory Risk" in result["risks"]

    def test_match_cybersecurity_risk(self, service):
        """Test matching Cybersecurity Risk."""
        summary = "The data breach affected customer security and privacy"
        result = service.structure_thesis(summary)

        assert "Cybersecurity Risk" in result["risks"]

    def test_match_investment_signal_ai_driven(self, service):
        """Test matching AI-Driven Financial Tools investment signal."""
        summary = "Our AI chatbot automates financial advisory services"
        result = service.structure_thesis(summary)

        assert "AI-Driven Financial Tools" in result["investment_signals"]

    # === Multiple Match Tests ===

    def test_multiple_themes_ranked_by_keyword_hits(self, service):
        """Test that categories with more keyword hits are included."""
        summary = "Digital banking neobank challenger bank online banking enables payment transfers and p2p transactions"
        result = service.structure_thesis(summary)

        assert len(result["key_themes"]) >= 2
        assert "Neobanking" in result["key_themes"]
        assert "Digital Payments" in result["key_themes"]

    def test_returns_top_3_categories_max(self, service):
        """Test that at most 3 categories are returned."""
        summary = (
            "digital bank neobank payment transfer peer-to-peer blockchain crypto "
            "lending loan borrowing wealth robo-advisor portfolio insurance underwriting"
        )
        result = service.structure_thesis(summary)

        assert len(result["key_themes"]) <= 3
        assert len(result["risks"]) <= 3
        assert len(result["investment_signals"]) <= 3

    # === Edge Cases ===

    def test_empty_summary(self, service):
        """Test with empty summary."""
        result = service.structure_thesis("")

        assert result["key_themes"] == []
        assert result["risks"] == []
        assert result["investment_signals"] == []

    def test_no_matching_keywords(self, service):
        """Test with summary containing no recognized keywords."""
        summary = "The weather is nice and the food was delicious"
        result = service.structure_thesis(summary)

        assert result["key_themes"] == []
        assert result["risks"] == []
        assert result["investment_signals"] == []

    def test_case_insensitive_matching(self, service):
        """Test that matching is case-insensitive."""
        result1 = service.structure_thesis("digital BANK")
        result2 = service.structure_thesis("DIGITAL bank")
        result3 = service.structure_thesis("DiGiTaL bAnK")

        assert result1["key_themes"] == result2["key_themes"] == result3["key_themes"]
        assert "Neobanking" in result1["key_themes"]

    def test_substring_keyword_matching(self, service):
        """Test that keywords match as substrings in text."""
        result = service.structure_thesis("We offer digital banking solutions")

        assert "Neobanking" in result["key_themes"]

    # === Category Map Tests (test data directly, not via service) ===

    def test_all_theme_categories_defined(self):
        """Test that ThemeMappings has expected categories."""
        expected_themes = [
            "AI-Powered Automation",
            "Digital Payments",
            "Blockchain & Web3",
            "Digital Lending",
            "Neobanking",
            "WealthTech",
            "B2B Finance",
            "RegTech & Compliance",
            "Embedded Finance",
            "Consumer Finance",
            "Fintech Infrastructure",
            "Insurtech",
        ]
        assert set(ThemeMappings.get_mapping().categories.keys()) == set(expected_themes)

    def test_all_risk_categories_defined(self):
        """Test that RiskMappings has expected categories."""
        expected_risks = [
            "Regulatory Risk",
            "Cybersecurity Risk",
            "Market Adoption Risk",
            "Competitive Pressure",
            "Credit & Liquidity Risk",
            "Macroeconomic Risk",
            "Data Privacy Risk",
            "Scalability Risk",
            "Geopolitical Risk",
            "Concentration Risk",
        ]
        assert set(RiskMappings.get_mapping().categories.keys()) == set(expected_risks)

    def test_all_signal_categories_defined(self):
        """Test that SignalMappings has expected categories."""
        expected_signals = [
            "B2B Fintech Expansion",
            "AI-Driven Financial Tools",
            "Emerging Market Growth",
            "Payment Infrastructure",
            "Embedded Finance Opportunity",
            "Consumer Fintech Adoption",
            "Alternative Lending Growth",
            "Crypto & Web3 Opportunity",
            "RegTech Investment Signal",
            "WealthTech Disruption",
        ]
        assert set(SignalMappings.get_mapping().categories.keys()) == set(expected_signals)

    # === _match_categories Tests ===

    def test_match_categories_with_no_matches(self, service):
        """Test _match_categories with text containing no keywords."""
        text = "the weather is nice"
        category_map = {"Category A": ["keyword1", "keyword2"]}
        result = service._match_categories(text, category_map)

        assert result == []

    def test_match_categories_single_match(self, service):
        """Test _match_categories with single category matched."""
        text = "keyword1 appears here"
        category_map = {
            "Category A": ["keyword1", "keyword2"],
            "Category B": ["keyword3"],
        }
        result = service._match_categories(text, category_map)

        assert result == ["Category A"]

    def test_match_categories_multiple_hits_rank_higher(self, service):
        """Test that categories with more keyword hits rank higher."""
        text = "keyword1 keyword2 keyword3"
        category_map = {
            "Category A": ["keyword1", "keyword2"],  # 2 keyword matches
            "Category B": ["keyword3"],              # 1 keyword match
        }
        result = service._match_categories(text, category_map)

        assert result[0] == "Category A"

    def test_match_categories_ranking_by_hits(self, service):
        """Test that categories are ranked by number of keyword hits."""
        text = "a b c d e"
        category_map = {
            "Category A": ["a"],       # 1 hit
            "Category B": ["b", "c"],  # 2 hits
            "Category C": ["d", "e"],  # 2 hits
        }
        result = service._match_categories(text, category_map)

        assert "Category A" not in result[:2]
        assert result[2] == "Category A"

    def test_match_categories_respects_max_results(self, service):
        """Test that _match_categories returns at most max_results results."""
        text = "a b c d e"
        category_map = {
            "Cat1": ["a"],
            "Cat2": ["b"],
            "Cat3": ["c"],
            "Cat4": ["d"],
            "Cat5": ["e"],
        }
        result = service._match_categories(text, category_map)

        assert len(result) == 3  # default max_results=3

    # === Real-World Scenario Tests ===

    def test_realistic_fintech_summary(self, service):
        """Test with realistic fintech article summary."""
        summary = (
            "Stripe launches embedded payments API for e-commerce platforms. "
            "The new API enables merchants to integrate payment processing directly into their apps. "
            "This is part of Stripe's strategy to dominate embedded finance and become the infrastructure "
            "backbone for the fintech ecosystem. The move signals strong investment opportunity in "
            "payment infrastructure and API-first platforms."
        )
        result = service.structure_thesis(summary)

        assert any(theme in result["key_themes"] for theme in [
            "Digital Payments", "Embedded Finance", "Fintech Infrastructure"
        ])

    def test_realistic_risk_summary(self, service):
        """Test risk detection in realistic summary."""
        summary = (
            "Crypto exchange faces SEC regulatory scrutiny and data breach that exposed customer records. "
            "The security incident raises concerns about the company's infrastructure and compliance practices."
        )
        result = service.structure_thesis(summary)

        assert "Regulatory Risk" in result["risks"]
        assert "Cybersecurity Risk" in result["risks"]
