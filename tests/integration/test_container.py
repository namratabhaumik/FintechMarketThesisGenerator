"""Integration tests for dependency injection container."""

import pytest

from config.settings import AppConfig
from dependency_injection.container import ServiceContainer


class TestServiceContainer:
    """Tests for ServiceContainer."""

    def test_container_initialization(self):
        """Test that container initializes without errors."""
        config = AppConfig()
        config.llm.api_key = "test_key"  # Mock API key

        container = ServiceContainer(config)
        assert container is not None

    def test_get_scraper(self):
        """Test getting scraper from container."""
        config = AppConfig()
        container = ServiceContainer(config)

        scraper = container.get_scraper()
        assert scraper is not None
        assert callable(getattr(scraper, "scrape"))

    def test_get_services(self):
        """Test getting services from container."""
        config = AppConfig()
        config.llm.api_key = "test_key"
        config.embedding.provider = "huggingface"  # Use HuggingFace for testing

        container = ServiceContainer(config)

        ingestion = container.get_ingestion_service()
        assert ingestion is not None

        # Note: Other services require actual models, so we skip those
        pytest.skip("Other services require actual model initialization")

    def test_container_singleton_caching(self):
        """Test that container caches instances."""
        config = AppConfig()
        container = ServiceContainer(config)

        scraper1 = container.get_scraper()
        scraper2 = container.get_scraper()

        assert scraper1 is scraper2  # Same instance
