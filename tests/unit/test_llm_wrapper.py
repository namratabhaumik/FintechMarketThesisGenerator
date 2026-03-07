"""Unit tests for LLMWrapper retry and fallback functionality."""

import pytest
from unittest.mock import Mock, patch
from langchain.docstore.document import Document

from core.implementations.llm.llm_wrapper import LLMWrapper
from core.interfaces.llm import ILanguageModel


@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        Document(page_content="Test content 1", metadata={"url": "http://test1.com"}),
        Document(page_content="Test content 2", metadata={"url": "http://test2.com"}),
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

        result = wrapper.summarize(test_documents)

        assert result == "Primary summary"
        assert mock_primary_llm.summarize.call_count == 1
        assert mock_fallback_llm.summarize.call_count == 0

    def test_model_name_includes_both_llms(self, mock_primary_llm, mock_fallback_llm):
        """get_model_name shows both primary and fallback."""
        wrapper = LLMWrapper(
            primary_llm=mock_primary_llm,
            fallback_llm=mock_fallback_llm
        )

        model_name = wrapper.get_model_name()
        assert "primary-llm" in model_name
        assert "fallback-llm" in model_name


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

        result = wrapper.summarize(test_documents)

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
            wrapper.summarize(test_documents)

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

        result = wrapper.summarize(test_documents)

        assert result == "Success on attempt 3"
        assert mock_primary_llm.summarize.call_count == 3
        assert mock_fallback_llm.summarize.call_count == 0

    @patch('time.sleep')
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

        wrapper.summarize(test_documents)

        # Should sleep with exponential backoff: 1.0s, then 2.0s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

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

        wrapper.summarize(test_documents)

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

        result = wrapper.summarize(test_documents)

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

        result = wrapper.summarize([])

        assert result == "Empty summary"
        mock_primary_llm.summarize.assert_called_once_with([])

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

        result = wrapper.summarize(test_documents)

        assert result == "Fallback summary"
        assert mock_primary_llm.summarize.call_count == 3
