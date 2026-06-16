"""Unit tests for Article and StructuredThesis data models."""

from datetime import datetime, timezone

import pytest
from core.models.article import Article
from core.models.raw_article import RawArticle
from core.models.thesis import StructuredThesis

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestArticle:
    """Tests for Article dataclass validation and creation."""

    # === Valid Creation ===

    def test_valid_article_creation(self):
        article = Article(title="Test", text="Some content here.", source="example.com", published_at=PUB)
        assert article.title == "Test"
        assert article.text == "Some content here."
        assert article.source == "example.com"
        assert article.published_at == PUB

    def test_article_url_defaults_to_none(self):
        article = Article(title="Test", text="Some content.", source="example.com", published_at=PUB)
        assert article.url is None

    def test_article_with_url(self):
        article = Article(
            title="Test", text="Content.", source="example.com",
            url="https://example.com/article", published_at=PUB
        )
        assert article.url == "https://example.com/article"

    # === Validation: title ===

    def test_empty_title_raises(self):
        with pytest.raises(ValueError, match="title"):
            Article(title="", text="Content.", source="example.com", published_at=PUB)

    def test_whitespace_only_title_raises(self):
        with pytest.raises(ValueError, match="title"):
            Article(title="   ", text="Content.", source="example.com", published_at=PUB)

    # === Validation: text ===

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            Article(title="Title", text="", source="example.com", published_at=PUB)

    def test_whitespace_only_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            Article(title="Title", text="   ", source="example.com", published_at=PUB)

    # === Validation: source ===

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            Article(title="Title", text="Content.", source="", published_at=PUB)

    def test_whitespace_only_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            Article(title="Title", text="Content.", source="   ", published_at=PUB)

    # === Validation: published_at ===

    def test_missing_published_at_raises(self):
        with pytest.raises(TypeError):
            Article(title="Title", text="Content.", source="example.com")

    def test_non_datetime_published_at_raises(self):
        with pytest.raises(ValueError, match="published_at"):
            Article(title="Title", text="Content.", source="example.com", published_at="2026-01-01")


class TestRawArticle:
    """Tests for RawArticle (Bronze) validation and defaults."""

    def test_valid_creation_with_defaults(self):
        raw = RawArticle(title="T", url="https://x/1", published_at=PUB)
        assert raw.url == "https://x/1"
        assert raw.summary == ""
        assert raw.source == ""
        assert raw.feed_name == ""

    def test_empty_title_raises(self):
        with pytest.raises(ValueError, match="title"):
            RawArticle(title="  ", url="https://x/1", published_at=PUB)

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="url"):
            RawArticle(title="T", url="", published_at=PUB)

    def test_non_datetime_published_at_raises(self):
        with pytest.raises(ValueError, match="published_at"):
            RawArticle(title="T", url="https://x/1", published_at="2026-01-01")


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
