"""Pytest configuration and shared fixtures."""

import pytest
from typing import List

from core.interfaces.article_source import IArticleSource
from core.interfaces.scraper import IWebScraper
from core.interfaces.llm import ILanguageModel
from core.models.article import Article


# === Mock Implementations ===

class MockWebScraper(IWebScraper):
    """Mock scraper for testing."""

    def scrape(self, url: str) -> str:
        """Return mock content."""
        return f"Mock content from {url}"


class MockArticleSource(IArticleSource):
    """Mock article source for testing."""

    def __init__(self, articles: List[Article]):
        """Initialize with predefined articles."""
        self._articles = articles

    def fetch_articles(self, query: str, limit: int) -> List[Article]:
        """Return predefined articles."""
        return self._articles[:limit]

    def get_source_name(self) -> str:
        """Return source name."""
        return "Mock Source"


class MockLanguageModel(ILanguageModel):
    """Mock LLM for testing."""

    def summarize(self, documents) -> str:
        """Return mock summary."""
        return "Mock summary: " + " ".join([d.page_content[:50] for d in documents])

    def get_model_name(self) -> str:
        """Return model name."""
        return "mock-model"


# === Fixtures ===

@pytest.fixture
def sample_articles() -> List[Article]:
    """Sample articles for testing."""
    return [
        Article(
            title="Test Article 1",
            text="This is test content about fintech innovations and digital banking.",
            source="example.com",
            url="https://example.com/article1"
        ),
        Article(
            title="Test Article 2",
            text="This is test content about blockchain and cryptocurrency trends.",
            source="example.com",
            url="https://example.com/article2"
        ),
        Article(
            title="Test Article 3",
            text="This is test content about payment systems and financial services.",
            source="techcrunch.com",
            url="https://techcrunch.com/article3"
        )
    ]


@pytest.fixture
def mock_scraper() -> IWebScraper:
    """Mock web scraper."""
    return MockWebScraper()


@pytest.fixture
def mock_article_source(sample_articles) -> IArticleSource:
    """Mock article source."""
    return MockArticleSource(sample_articles)


@pytest.fixture
def mock_llm() -> ILanguageModel:
    """Mock language model."""
    return MockLanguageModel()


