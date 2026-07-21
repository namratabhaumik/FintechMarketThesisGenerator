"""Tests for AI Gateway routing components."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from langchain_core.documents import Document

from core.implementations.llm.cache_manager import CacheManager
from core.implementations.llm.usage_tracker import UsageTracker
from core.implementations.llm.ai_gateway import AIGateway
from core.implementations.llm.routing_strategy import (
    ROUTE_FALLBACK,
    ROUTE_PRIMARY,
    CostOptimizedStrategy,
    QualityFirstStrategy,
    HybridStrategy,
    get_strategy,
)
from config.settings import AIGatewayConfig


class TestCacheManager:
    """Tests for CacheManager."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        manager = CacheManager()
        key1 = manager.generate_key("content1", "topic1", "model1")
        key2 = manager.generate_key("content1", "topic1", "model1")

        # Same inputs should generate same key
        assert key1 == key2

        # Different inputs should generate different keys
        key3 = manager.generate_key("content2", "topic1", "model1")
        assert key1 != key3

    def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        manager = CacheManager()
        key = manager.generate_key("content", "topic", "model")

        manager.set(key, "response text", "model-name", 100, 50)
        entry = manager.get(key)

        assert entry is not None
        assert entry.response == "response text"
        assert entry.model == "model-name"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50

    def test_cache_miss(self):
        """Test cache miss returns None."""
        manager = CacheManager()
        entry = manager.get("non_existent_key")
        assert entry is None

    def test_cache_expiration(self):
        """Test cache expiration based on TTL."""
        manager = CacheManager(ttl_seconds=1)
        key = manager.generate_key("content", "topic", "model")

        manager.set(key, "response", "model", 100, 50)
        entry = manager.get(key)
        assert entry is not None

        # Simulate expiration
        manager._cache[key].created_at = datetime.now() - __import__('datetime').timedelta(seconds=2)
        entry = manager.get(key)
        assert entry is None

    def test_cache_metrics(self):
        """Test cache performance metrics."""
        manager = CacheManager()
        key = manager.generate_key("content", "topic", "model")

        manager.set(key, "response", "model", 100, 50)

        # First hit
        manager.get(key)
        # Second miss (different key)
        manager.get("different_key")

        metrics = manager.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == 50.0
        assert metrics["cache_size"] == 1

    def test_cache_clear(self):
        """Test cache clearing (asserted through the get() contract)."""
        manager = CacheManager()
        key = manager.generate_key("content", "topic", "model")
        manager.set(key, "response", "model", 100, 50)

        assert manager.get(key) is not None
        manager.clear()
        assert manager.get(key) is None

    def test_cache_hit_restores_local_provenance(self):
        """A cached local-extractive summary stays marked 'local' on later hits
        (no model runs on the cache path, so the gateway must restore it)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from config.settings import AIGatewayConfig
        from core.implementations.llm.ai_gateway import AIGateway
        from core.implementations.llm.usage_tracker import UsageTracker
        from core.interfaces.llm import SOURCE_LLM, summary_source_var
        from langchain_core.documents import Document

        local_llm = MagicMock()
        local_llm.get_model_name.return_value = "local-extractor"
        local_llm.summarize = AsyncMock(return_value="extractive summary")

        gateway = AIGateway(
            primary_llm=local_llm,
            fallback_llm=MagicMock(),
            config=AIGatewayConfig(enabled=True),
            cache_manager=CacheManager(),
            usage_tracker=UsageTracker(),
        )
        docs = [Document(page_content="x", metadata={"url": "u"})]

        async def scenario():
            # Read the var inside the task: context changes don't propagate
            # out of asyncio.run.
            await gateway.summarize(docs)  # miss: caches with model name
            summary_source_var.set(SOURCE_LLM)  # simulate a fresh request
            await gateway.summarize(docs)  # hit: must restore provenance
            return summary_source_var.get()

        source = asyncio.run(scenario())

        assert local_llm.summarize.call_count == 1
        assert source == "local"


class TestUsageTracker:
    """Tests for UsageTracker (call counts / tokens, no dollars)."""

    def test_record_call_returns_daily_primary_count(self):
        """A real primary call increments and returns the daily primary count."""
        tracker = UsageTracker()

        count = tracker.record_call(
            provider="primary",
            model="gemini-x",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1500.0,
        )

        assert count == 1
        metrics = tracker.get_metrics()
        assert metrics["total_calls"] == 1
        assert metrics["daily_primary_calls"] == 1

    def test_cache_hits_and_local_do_not_consume_budget(self):
        """Cache hits and local-summarizer calls don't count against budget."""
        tracker = UsageTracker()

        tracker.record_call("primary", "gemini-x", 1000, 500, 50.0, cache_hit=True)
        tracker.record_call("local", "local-extractor", 0, 0, 100.0)

        assert tracker.get_daily_calls() == 0
        metrics = tracker.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["daily_primary_calls"] == 0

    def test_daily_calls_counts_only_billable(self):
        """Daily calls counts real primary calls only."""
        tracker = UsageTracker()

        tracker.record_call("primary", "gemini-x", 1000, 500, 1000.0)
        tracker.record_call("primary", "gemini-x", 1000, 500, 1000.0)
        tracker.record_call("local", "local-extractor", 0, 0, 100.0)

        assert tracker.get_daily_calls() == 2

    def test_metrics_aggregation_by_provider(self):
        """Test metrics aggregation by provider."""
        tracker = UsageTracker()

        tracker.record_call("primary", "gemini-x", 1000, 500, 1000.0)
        tracker.record_call("local", "local-extractor", 1000, 500, 100.0)

        metrics = tracker.get_metrics()
        assert "primary" in metrics["by_provider"]
        assert "local" in metrics["by_provider"]
        assert metrics["by_provider"]["primary"]["calls"] == 1
        assert metrics["by_provider"]["local"]["calls"] == 1


class TestRoutingStrategies:
    """Tests for routing strategies (size + daily call budget)."""

    def test_cost_optimized_strategy_large_documents(self):
        """Test cost-optimized routing for large documents."""
        strategy = CostOptimizedStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content=" ".join(["word"] * 4000))]  # ~5200 tokens

        assert strategy.select_route(docs, "topic", 0, 50) == ROUTE_FALLBACK

    def test_cost_optimized_strategy_call_budget(self):
        """Test cost-optimized routing when approaching the call budget."""
        strategy = CostOptimizedStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]  # ~10 tokens

        # 90% of budget (>80%) - should route to the local summarizer
        assert strategy.select_route(docs, "topic", 45, 50) == ROUTE_FALLBACK

    def test_cost_optimized_strategy_small_documents(self):
        """Test cost-optimized routing for small documents within budget."""
        strategy = CostOptimizedStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]  # ~10 tokens

        # 50% of budget - should route to the primary LLM
        assert strategy.select_route(docs, "topic", 25, 50) == ROUTE_PRIMARY

    def test_quality_first_strategy_always_gemini(self):
        """Test quality-first strategy always uses the primary LLM."""
        strategy = QualityFirstStrategy()
        docs = [Document(page_content="any content")]

        assert strategy.select_route(docs, "topic", 49, 50) == ROUTE_PRIMARY

    def test_hybrid_strategy_hard_call_budget(self):
        """Test hybrid strategy enforces the hard daily call budget."""
        strategy = HybridStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]

        # Call budget reached - hard constraint
        assert strategy.select_route(docs, "topic", 50, 50) == ROUTE_FALLBACK

    def test_hybrid_strategy_large_documents(self):
        """Test hybrid strategy routes large docs to local."""
        strategy = HybridStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content=" ".join(["word"] * 4000))]  # ~5200 tokens

        assert strategy.select_route(docs, "topic", 1, 50) == ROUTE_FALLBACK

    def test_hybrid_strategy_small_documents_in_budget(self):
        """Test hybrid strategy routes small docs to the primary LLM when in budget."""
        strategy = HybridStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]

        assert strategy.select_route(docs, "topic", 1, 50) == ROUTE_PRIMARY

    def test_get_strategy_factory(self):
        """Test strategy factory function."""
        strategy = get_strategy("cost_optimized")
        assert isinstance(strategy, CostOptimizedStrategy)

        strategy = get_strategy("quality_first")
        assert isinstance(strategy, QualityFirstStrategy)

        strategy = get_strategy("hybrid")
        assert isinstance(strategy, HybridStrategy)

    def test_get_strategy_invalid(self):
        """Test factory raises error for unknown strategy."""
        with pytest.raises(ValueError):
            get_strategy("unknown_strategy")


class TestAIGateway:
    """Tests for AI Gateway wrapper."""

    @pytest.fixture
    def mock_llms(self):
        """Create mock LLM instances."""
        primary = Mock()
        primary.get_model_name.return_value = "gemini-x"
        primary.summarize = AsyncMock(return_value="Primary summary")

        fallback = Mock()
        fallback.get_model_name.return_value = "local-extractor"
        fallback.summarize = AsyncMock(return_value="Fallback summary")

        return primary, fallback

    @pytest.fixture
    def gateway_config(self):
        """Create AI Gateway config."""
        return AIGatewayConfig(
            enabled=True,
            strategy="hybrid",
            cache_enabled=True,
            cache_ttl_seconds=3600,
            call_budget_daily=50,
            track_metrics=True,
        )

    def test_gateway_initialization(self, mock_llms, gateway_config):
        """Test AI Gateway initialization."""
        primary, fallback = mock_llms

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=CacheManager(),
            usage_tracker=UsageTracker(),
        )

        assert gateway is not None
        assert "AIGateway" in gateway.get_model_name()

    def test_gateway_cache_hit(self, mock_llms, gateway_config):
        """Test gateway returns cached result on cache hit."""
        primary, fallback = mock_llms
        cache_manager = CacheManager()

        # Pre-populate cache
        cache_key = cache_manager.generate_key("content", "fintech", "combined")
        cache_manager.set(cache_key, "Cached response", "model", 100, 50)

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            usage_tracker=UsageTracker(),
        )

        docs = [Document(page_content="content")]
        result = asyncio.run(gateway.summarize(docs, "fintech"))

        assert result == "Cached response"
        # Primary should not be called on cache hit
        primary.summarize.assert_not_called()

    def test_gateway_calls_primary_on_cache_miss(self, mock_llms, gateway_config):
        """Test gateway calls primary LLM on cache miss."""
        primary, fallback = mock_llms

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=CacheManager(),
            usage_tracker=UsageTracker(),
        )

        docs = [Document(page_content="new content")]
        result = asyncio.run(gateway.summarize(docs))

        assert result == "Primary summary"
        primary.summarize.assert_called_once()

    def test_gateway_fallback_on_primary_failure(self, mock_llms, gateway_config):
        """Test gateway falls back to secondary on primary failure."""
        primary, fallback = mock_llms
        primary.summarize.side_effect = Exception("Primary failed")

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=CacheManager(),
            usage_tracker=UsageTracker(),
        )

        docs = [Document(page_content="content")]
        result = asyncio.run(gateway.summarize(docs))

        assert result == "Fallback summary"
        fallback.summarize.assert_called_once()

    def test_gateway_usage_tracking(self, mock_llms, gateway_config):
        """Test gateway tracks usage."""
        primary, fallback = mock_llms
        usage_tracker = UsageTracker()

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=CacheManager(),
            usage_tracker=usage_tracker,
        )

        docs = [Document(page_content="content " * 100)]
        asyncio.run(gateway.summarize(docs))

        metrics = usage_tracker.get_metrics()
        assert metrics["total_calls"] > 0

    def test_gateway_call_budget_forces_fallback(self, mock_llms, gateway_config):
        """Past the daily call budget, the gateway routes to the local summarizer."""
        primary, fallback = mock_llms
        usage_tracker = UsageTracker()
        # Saturate the budget with real primary calls.
        for _ in range(gateway_config.call_budget_daily):
            usage_tracker.record_call("primary", "gemini-x", 100, 50, 10.0)

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=CacheManager(),
            usage_tracker=usage_tracker,
        )

        docs = [Document(page_content="fresh content")]
        result = asyncio.run(gateway.summarize(docs))

        assert result == "Fallback summary"
        primary.summarize.assert_not_called()
        fallback.summarize.assert_called_once()

    def test_gateway_metrics(self, mock_llms, gateway_config):
        """Test gateway metrics aggregation."""
        primary, fallback = mock_llms

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=CacheManager(),
            usage_tracker=UsageTracker(),
        )

        metrics = gateway.get_metrics()
        assert "cache" in metrics
        assert "usage" in metrics
        assert metrics["gateway_enabled"] is True
        assert metrics["strategy"] == "hybrid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
