"""Unit tests for Article and StructuredThesis data models."""

import pytest
from core.models.article import Article
from core.models.thesis import StructuredThesis


class TestArticle:
    """Tests for Article dataclass validation and creation."""

    # === Valid Creation ===

    def test_valid_article_creation(self):
        article = Article(title="Test", text="Some content here.", source="example.com")
        assert article.title == "Test"
        assert article.text == "Some content here."
        assert article.source == "example.com"

    def test_article_url_defaults_to_none(self):
        article = Article(title="Test", text="Some content.", source="example.com")
        assert article.url is None

    def test_article_with_url(self):
        article = Article(
            title="Test", text="Content.", source="example.com",
            url="https://example.com/article"
        )
        assert article.url == "https://example.com/article"

    # === Validation: title ===

    def test_empty_title_raises(self):
        with pytest.raises(ValueError, match="title"):
            Article(title="", text="Content.", source="example.com")

    def test_whitespace_only_title_raises(self):
        with pytest.raises(ValueError, match="title"):
            Article(title="   ", text="Content.", source="example.com")

    # === Validation: text ===

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            Article(title="Title", text="", source="example.com")

    def test_whitespace_only_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            Article(title="Title", text="   ", source="example.com")

    # === Validation: source ===

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            Article(title="Title", text="Content.", source="")

    def test_whitespace_only_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            Article(title="Title", text="Content.", source="   ")


class TestStructuredThesis:
    """Tests for StructuredThesis dataclass defaults and creation."""

    def test_default_fields_are_empty_lists(self):
        thesis = StructuredThesis()
        assert thesis.key_themes == []
        assert thesis.risks == []
        assert thesis.investment_signals == []
        assert thesis.sources == []

    def test_raw_output_defaults_to_none(self):
        assert StructuredThesis().raw_output is None

    def test_creation_with_data(self):
        thesis = StructuredThesis(
            key_themes=["Digital Payments"],
            risks=["Regulatory Risk"],
            investment_signals=["AI-Driven Financial Tools"],
            sources=["https://example.com"],
            raw_output="Summary text."
        )
        assert thesis.key_themes == ["Digital Payments"]
        assert thesis.risks == ["Regulatory Risk"]
        assert thesis.investment_signals == ["AI-Driven Financial Tools"]
        assert thesis.sources == ["https://example.com"]
        assert thesis.raw_output == "Summary text."

    def test_default_lists_are_independent(self):
        """Default factory ensures each instance gets its own list."""
        t1 = StructuredThesis()
        t2 = StructuredThesis()
        t1.key_themes.append("Digital Payments")
        assert t2.key_themes == []
