"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timezone
from typing import Dict, List

from core.interfaces.scraper import IWebScraper
from core.interfaces.llm import ILanguageModel
from core.interfaces.scoring_strategy import IScoringStrategy
from core.models.article import Article


# === Mock Implementations ===

class MockWebScraper(IWebScraper):
    """Mock scraper for testing."""

    def scrape(self, url: str) -> str:
        """Return mock content."""
        return f"Mock content from {url}"


class MockLanguageModel(ILanguageModel):
    """Mock LLM for testing."""

    async def summarize(self, documents) -> str:
        """Return mock summary."""
        return "Mock summary: " + " ".join([d.page_content[:50] for d in documents])

    def get_model_name(self) -> str:
        """Return model name."""
        return "mock-model"


class MockScoringStrategy(IScoringStrategy):
    """Mock scoring strategy for testing."""

    def score(self, text: str, category_keywords: Dict[str, List[str]]) -> Dict[str, int]:
        """Return mock scores."""
        return {
            label: len(keywords)
            for label, keywords in category_keywords.items()
        }


# === Fixtures ===

@pytest.fixture
def sample_articles() -> List[Article]:
    """Sample articles for testing."""
    return [
        Article(
            title="Test Article 1",
            text="This is test content about fintech innovations and digital banking.",
            source="example.com",
            url="https://example.com/article1",
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        Article(
            title="Test Article 2",
            text="This is test content about blockchain and cryptocurrency trends.",
            source="example.com",
            url="https://example.com/article2",
            published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        Article(
            title="Test Article 3",
            text="This is test content about payment systems and financial services.",
            source="techcrunch.com",
            url="https://techcrunch.com/article3",
            published_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
    ]


@pytest.fixture
def mock_scraper() -> IWebScraper:
    """Mock web scraper."""
    return MockWebScraper()


@pytest.fixture
def mock_llm() -> ILanguageModel:
    """Mock language model."""
    return MockLanguageModel()


@pytest.fixture
def mock_scoring_strategy() -> IScoringStrategy:
    """Mock scoring strategy."""
    return MockScoringStrategy()
