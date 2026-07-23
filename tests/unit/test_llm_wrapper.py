"""Unit tests for LLMWrapper retry and fallback functionality."""

import asyncio

import pytest
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.documents import Document

from core.implementations.llm.llm_wrapper import LLMWrapper
from core.interfaces.llm import ILanguageModel


@pytest.fixture
def test_documents():
    """Create test documents.

    The page content is real, extractable prose (>20 chars, sentence-final
    punctuation) so tests that exercise the real local summarizer fallback get a
    non-empty extractive summary; the mock-LLM tests ignore the content.
    """
    return [
        Document(
            page_content="Fintech adoption is accelerating across emerging markets this year.",
            metadata={"url": "http://test1.com"},
        ),
        Document(
            page_content="Digital payment platforms are attracting significant investment momentum.",
            metadata={"url": "http://test2.com"},
        ),
    ]


@pytest.fixture
def mock_primary_llm():
    """Create a mock primary LLM."""
    mock = Mock(spec=ILanguageModel)
    mock.get_model_name.return_value = "primary-llm"
    return mock


@pytest.fixture
def mock_fallback_llm():
    """Create a mock fallback LLM."""
    mock = Mock(spec=ILanguageModel)
    mock.get_model_name.return_value = "fallback-llm"
    return mock


class TestLLMWrapperSuccess:
    """Test successful primary LLM execution."""

    def test_primary_succeeds_on_first_attempt(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """Primary LLM returns summary on first attempt."""
        mock_primary_llm.summarize.return_value = "Primary summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=2
        )

        result = asyncio.run(wrapper.summarize(test_documents))

        assert result == "Primary summary"
        assert mock_primary_llm.summarize.call_count == 1
        assert mock_fallback_llm.summarize.call_count == 0

    def test_model_name_returns_primary_only(self, mock_primary_llm, mock_fallback_llm):
        """get_model_name returns the primary (billable) model name only."""
        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm
        )

        model_name = wrapper.get_model_name()
        assert "primary-llm" in model_name
        assert "fallback-llm" not in model_name


class TestLLMWrapperFallback:
    """Test fallback behavior when primary fails."""

    def test_fallback_on_primary_failure(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """Falls back to secondary when primary fails."""
        mock_primary_llm.summarize.side_effect = RuntimeError("API Error")
        mock_fallback_llm.summarize.return_value = "Fallback summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=1
        )

        result = asyncio.run(wrapper.summarize(test_documents))

        assert result == "Fallback summary"
        # 1 attempt + 1 retry = 2 calls to primary
        assert mock_primary_llm.summarize.call_count == 2
        assert mock_fallback_llm.summarize.call_count == 1

    def test_fallback_failure_raises_exception(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """Raises exception if both primary and fallback fail."""
        mock_primary_llm.summarize.side_effect = RuntimeError("Primary Error")
        mock_fallback_llm.summarize.side_effect = RuntimeError("Fallback Error")

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=1
        )

        with pytest.raises(RuntimeError, match="Fallback Error"):
            asyncio.run(wrapper.summarize(test_documents))

        assert mock_primary_llm.summarize.call_count == 2
        assert mock_fallback_llm.summarize.call_count == 1


class TestLLMWrapperRetry:
    """Test retry logic with exponential backoff."""

    def test_retries_until_success(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """Retries primary until it succeeds."""
        mock_primary_llm.summarize.side_effect = [
            RuntimeError("Attempt 1 failed"),
            RuntimeError("Attempt 2 failed"),
            "Success on attempt 3"
        ]

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=2,
            initial_delay_seconds=0.01
        )

        result = asyncio.run(wrapper.summarize(test_documents))

        assert result == "Success on attempt 3"
        assert mock_primary_llm.summarize.call_count == 3
        assert mock_fallback_llm.summarize.call_count == 0

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_exponential_backoff(self, mock_sleep, test_documents, mock_primary_llm, mock_fallback_llm):
        """Uses exponential backoff between retries."""
        mock_primary_llm.summarize.side_effect = RuntimeError("Always fails")
        mock_fallback_llm.summarize.return_value = "Fallback summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=2,
            initial_delay_seconds=1.0
        )

        asyncio.run(wrapper.summarize(test_documents))

        # Should sleep with exponential backoff: 1.0s, then 2.0s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    def test_retry_budget_stops_slow_hangs(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """A slow-failing primary (per-attempt timeout) exhausts the wall-clock
        budget after one attempt: no further retries even though max_retries
        allows them, so hangs cannot stack past a gateway timeout."""
        async def slow_failure(*args, **kwargs):
            await asyncio.sleep(0.1)  # longer than the whole budget below
            raise RuntimeError("hang then fail")

        mock_primary_llm.summarize.side_effect = slow_failure
        mock_fallback_llm.summarize.return_value = "Fallback summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=5,
            initial_delay_seconds=0.01,
            retry_budget_seconds=0.05,
        )

        result = asyncio.run(wrapper.summarize(test_documents))

        assert result == "Fallback summary"
        assert mock_primary_llm.summarize.call_count == 1

    def test_max_retries_respected(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """Respects max_retries configuration."""
        mock_primary_llm.summarize.side_effect = RuntimeError("Always fails")
        mock_fallback_llm.summarize.return_value = "Fallback summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=3,
            initial_delay_seconds=0.01
        )

        asyncio.run(wrapper.summarize(test_documents))

        # max_retries=3 means 1 initial attempt + 3 retries = 4 total attempts
        assert mock_primary_llm.summarize.call_count == 4


class TestLLMWrapperEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_retries(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """With max_retries=0, tries primary once then falls back."""
        mock_primary_llm.summarize.side_effect = RuntimeError("Fails")
        mock_fallback_llm.summarize.return_value = "Fallback summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=0
        )

        result = asyncio.run(wrapper.summarize(test_documents))

        assert result == "Fallback summary"
        assert mock_primary_llm.summarize.call_count == 1
        assert mock_fallback_llm.summarize.call_count == 1

    def test_empty_documents_list(self, mock_primary_llm, mock_fallback_llm):
        """Handles empty documents list."""
        mock_primary_llm.summarize.return_value = "Empty summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm
        )

        result = asyncio.run(wrapper.summarize([]))

        assert result == "Empty summary"
        mock_primary_llm.summarize.assert_called_once_with([], "")

    def test_fallback_summary_marks_local_provenance(self, test_documents, mock_primary_llm):
        """A summary served by the real local fallback flips the per-call
        provenance to 'local', so the thesis can record the degradation."""
        from config.settings import LLMConfig
        from core.implementations.llm.local_summarizer import LocalSummarizerModel
        from core.interfaces.llm import SOURCE_LLM, summary_source_var

        mock_primary_llm.summarize.side_effect = RuntimeError("API down")
        local = LocalSummarizerModel(LLMConfig(provider="local", model_name="x", api_key=""))
        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=local,
            max_retries=0,
        )

        async def run():
            # Read the var inside the task: context changes don't propagate
            # out of asyncio.run (in production the thesis service reads it
            # inside the same request task, like this).
            summary_source_var.set(SOURCE_LLM)
            result = await wrapper.summarize(test_documents)
            return result, summary_source_var.get()

        result, source = asyncio.run(run())

        assert result  # extractive text came back
        assert source == "local"

    def test_refine_raises_without_touching_fallback(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """refine retries the primary but never falls back (the local summarizer
        cannot rewrite a thesis); the error propagates to the caller so the API
        surfaces it instead of silently consuming a refinement round."""
        mock_primary_llm.refine.side_effect = RuntimeError("API down")

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=1,
            initial_delay_seconds=0.01,
        )

        with pytest.raises(RuntimeError, match="API down"):
            asyncio.run(wrapper.refine(test_documents, "thesis text", ["Too broad"]))

        # 1 attempt + 1 retry on primary; fallback never consulted.
        assert mock_primary_llm.refine.call_count == 2
        assert mock_fallback_llm.refine.call_count == 0

    def test_refine_not_implemented_raises_immediately(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """NotImplementedError from refine is re-raised at once - no retry, no fallback."""
        mock_primary_llm.refine.side_effect = NotImplementedError("no refine")

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=2,
            initial_delay_seconds=0.01,
        )

        with pytest.raises(NotImplementedError):
            asyncio.run(wrapper.refine(test_documents, "thesis text", ["Too broad"]))

        assert mock_primary_llm.refine.call_count == 1
        assert mock_fallback_llm.refine.call_count == 0

    def test_different_exception_types(self, test_documents, mock_primary_llm, mock_fallback_llm):
        """Handles different exception types from primary."""
        mock_primary_llm.summarize.side_effect = [
            ConnectionError("Network error"),
            TimeoutError("Timeout"),
            ValueError("Invalid response")
        ]
        mock_fallback_llm.summarize.return_value = "Fallback summary"

        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm,
            max_retries=2,
            initial_delay_seconds=0.01
        )

        result = asyncio.run(wrapper.summarize(test_documents))

        assert result == "Fallback summary"
        assert mock_primary_llm.summarize.call_count == 3
