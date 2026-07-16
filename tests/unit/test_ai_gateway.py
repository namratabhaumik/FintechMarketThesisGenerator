"""Tests for AI Gateway cost optimization components."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from langchain_core.documents import Document

from core.implementations.llm.cache_manager import CacheManager
from core.implementations.llm.cost_tracker import CostTracker
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
        from core.implementations.llm.cost_tracker import CostTracker
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
            cost_tracker=CostTracker(),
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


class TestCostTracker:
    """Tests for CostTracker."""

    def test_cost_calculation_gemini(self):
        """Test cost calculation for Gemini."""
        tracker = CostTracker()

        # Gemini 2.5 Flash pricing: $0.30 per 1M input, $2.50 per 1M output
        cost = tracker.calculate_cost("gemini", "gemini-2.5-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.80, rel=0.01)  # 0.30 + 2.50

    def test_cost_calculation_local(self):
        """Test cost calculation for local model (free)."""
        tracker = CostTracker()

        cost = tracker.calculate_cost("local", "local-extractor", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_cost_record_call(self):
        """Test recording API call cost."""
        tracker = CostTracker()

        cost = tracker.record_call(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1500.0,
        )

        assert cost > 0
        metrics = tracker.get_metrics()
        assert metrics["total_calls"] == 1
        assert metrics["daily_spend"] > 0

    def test_cache_hit_recording(self):
        """Test recording cache hits."""
        tracker = CostTracker()

        tracker.record_call(
            provider="gemini",
            model="gemini-2.0-flash",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=50.0,
            cache_hit=True,
        )

        metrics = tracker.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_hit_rate"] == 100.0

    def test_daily_spend_calculation(self):
        """Test daily spend calculation."""
        tracker = CostTracker()

        # Add a call for today
        tracker.record_call(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            latency_ms=1000.0,
        )

        daily_spend = tracker.get_daily_spend()
        assert daily_spend > 0

    def test_metrics_aggregation_by_provider(self):
        """Test metrics aggregation by provider."""
        tracker = CostTracker()

        tracker.record_call("gemini", "gemini-2.0-flash", 1000, 500, 1000.0)
        tracker.record_call("local", "local-extractor", 1000, 500, 100.0)

        metrics = tracker.get_metrics()
        assert "gemini" in metrics["by_provider"]
        assert "local" in metrics["by_provider"]
        assert metrics["by_provider"]["gemini"]["calls"] == 1
        assert metrics["by_provider"]["local"]["calls"] == 1


class TestRoutingStrategies:
    """Tests for routing strategies."""

    def test_cost_optimized_strategy_large_documents(self):
        """Test cost-optimized routing for large documents."""
        strategy = CostOptimizedStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content=" ".join(["word"] * 4000))]  # ~5200 tokens

        assert strategy.select_route(docs, "topic", 0, 5.0) == ROUTE_FALLBACK

    def test_cost_optimized_strategy_cost_limit(self):
        """Test cost-optimized routing when approaching cost limit."""
        strategy = CostOptimizedStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]  # ~10 tokens

        # 85% of limit - should route to the local summarizer
        assert strategy.select_route(docs, "topic", 4.25, 5.0) == ROUTE_FALLBACK

    def test_cost_optimized_strategy_small_documents(self):
        """Test cost-optimized routing for small documents within limit."""
        strategy = CostOptimizedStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]  # ~10 tokens

        # 50% of limit - should route to the primary LLM
        assert strategy.select_route(docs, "topic", 2.5, 5.0) == ROUTE_PRIMARY

    def test_quality_first_strategy_always_gemini(self):
        """Test quality-first strategy always uses the primary LLM."""
        strategy = QualityFirstStrategy()
        docs = [Document(page_content="any content")]

        assert strategy.select_route(docs, "topic", 4.9, 5.0) == ROUTE_PRIMARY

    def test_hybrid_strategy_hard_cost_limit(self):
        """Test hybrid strategy enforces hard cost limit."""
        strategy = HybridStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]

        # Cost limit exceeded - hard constraint
        assert strategy.select_route(docs, "topic", 5.0, 5.0) == ROUTE_FALLBACK

    def test_hybrid_strategy_large_documents(self):
        """Test hybrid strategy routes large docs to local."""
        strategy = HybridStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content=" ".join(["word"] * 4000))]  # ~5200 tokens

        assert strategy.select_route(docs, "topic", 1.0, 5.0) == ROUTE_FALLBACK

    def test_hybrid_strategy_small_documents_in_budget(self):
        """Test hybrid strategy routes small docs to the primary LLM when in budget."""
        strategy = HybridStrategy(size_threshold_tokens=5000)
        docs = [Document(page_content="small doc")]

        assert strategy.select_route(docs, "topic", 1.0, 5.0) == ROUTE_PRIMARY

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
        primary.get_model_name.return_value = "gemini-2.0-flash"
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
            cost_limit_daily=5.0,
            cost_limit_monthly=100.0,
            track_metrics=True,
        )

    def test_gateway_initialization(self, mock_llms, gateway_config):
        """Test AI Gateway initialization."""
        primary, fallback = mock_llms
        cache_manager = CacheManager()
        cost_tracker = CostTracker()

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            cost_tracker=cost_tracker,
        )

        assert gateway is not None
        assert "AIGateway" in gateway.get_model_name()

    def test_gateway_cache_hit(self, mock_llms, gateway_config):
        """Test gateway returns cached result on cache hit."""
        primary, fallback = mock_llms
        cache_manager = CacheManager()
        cost_tracker = CostTracker()

        # Pre-populate cache
        cache_key = cache_manager.generate_key("content", "fintech", "combined")
        cache_manager.set(cache_key, "Cached response", "model", 100, 50)

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            cost_tracker=cost_tracker,
        )

        docs = [Document(page_content="content")]
        result = asyncio.run(gateway.summarize(docs, "fintech"))

        assert result == "Cached response"
        # Primary should not be called on cache hit
        primary.summarize.assert_not_called()

    def test_gateway_calls_primary_on_cache_miss(self, mock_llms, gateway_config):
        """Test gateway calls primary LLM on cache miss."""
        primary, fallback = mock_llms
        cache_manager = CacheManager()
        cost_tracker = CostTracker()

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            cost_tracker=cost_tracker,
        )

        docs = [Document(page_content="new content")]
        result = asyncio.run(gateway.summarize(docs))

        assert result == "Primary summary"
        primary.summarize.assert_called_once()

    def test_gateway_fallback_on_primary_failure(self, mock_llms, gateway_config):
        """Test gateway falls back to secondary on primary failure."""
        primary, fallback = mock_llms
        primary.summarize.side_effect = Exception("Primary failed")

        cache_manager = CacheManager()
        cost_tracker = CostTracker()

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            cost_tracker=cost_tracker,
        )

        docs = [Document(page_content="content")]
        result = asyncio.run(gateway.summarize(docs))

        assert result == "Fallback summary"
        fallback.summarize.assert_called_once()

    def test_gateway_cost_tracking(self, mock_llms, gateway_config):
        """Test gateway tracks costs."""
        primary, fallback = mock_llms
        cache_manager = CacheManager()
        cost_tracker = CostTracker()

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            cost_tracker=cost_tracker,
        )

        docs = [Document(page_content="content " * 100)]
        asyncio.run(gateway.summarize(docs))

        metrics = cost_tracker.get_metrics()
        assert metrics["total_calls"] > 0

    def test_gateway_metrics(self, mock_llms, gateway_config):
        """Test gateway metrics aggregation."""
        primary, fallback = mock_llms
        cache_manager = CacheManager()
        cost_tracker = CostTracker()

        gateway = AIGateway(
            primary_llm=primary,
            fallback_llm=fallback,
            config=gateway_config,
            cache_manager=cache_manager,
            cost_tracker=cost_tracker,
        )

        metrics = gateway.get_metrics()
        assert "cache" in metrics
        assert "costs" in metrics
        assert metrics["gateway_enabled"] is True
        assert metrics["strategy"] == "hybrid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
