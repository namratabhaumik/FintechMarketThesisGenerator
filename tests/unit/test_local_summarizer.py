"""Unit tests for LocalSummarizerModel (pure-Python, no API calls)."""

import pytest
from unittest.mock import Mock
from langchain.docstore.document import Document

from config.settings import LLMConfig
from core.implementations.llm.local_summarizer import LocalSummarizerModel


@pytest.fixture
def config():
    """Minimal LLMConfig mock — API key unused by local model."""
    cfg = Mock(spec=LLMConfig)
    cfg.provider = "local"
    cfg.model_name = "local"
    cfg.api_key = ""
    cfg.temperature = 0.0
    return cfg


@pytest.fixture
def model(config):
    return LocalSummarizerModel(config)


def _doc(text: str) -> Document:
    """Helper to create a Document with no metadata."""
    return Document(page_content=text, metadata={})


# ─── get_model_name ──────────────────────────────────────────────────────────

class TestGetModelName:
    def test_returns_local_extractor(self, model):
        assert model.get_model_name() == "local-extractor"


# ─── _split_sentences ────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self, model):
        text = "Stripe raised $1 billion. The company plans global expansion."
        sentences = model._split_sentences(text)
        assert len(sentences) == 2

    def test_filters_short_sentences(self, model):
        """Sentences under 20 chars are dropped."""
        text = "Hi. Fintech payments are growing rapidly across digital platforms."
        sentences = model._split_sentences(text)
        assert all(len(s) > 20 for s in sentences)

    def test_requires_terminal_punctuation(self, model):
        """Sentences not ending in .!? are dropped."""
        text = "Stripe raised $1 billion in Series H funding"  # no terminal punctuation
        sentences = model._split_sentences(text)
        assert sentences == []

    def test_empty_text_returns_empty_list(self, model):
        assert model._split_sentences("") == []

    def test_exclamation_and_question_marks_accepted(self, model):
        text = "Is blockchain the future of finance? Absolutely it is transforming payments!"
        sentences = model._split_sentences(text)
        assert len(sentences) == 2


# ─── _score_sentence ─────────────────────────────────────────────────────────

class TestScoreSentence:
    def test_no_keywords_returns_zero(self, model):
        assert model._score_sentence("The weather is sunny today.") == 0

    def test_single_keyword_returns_one(self, model):
        assert model._score_sentence("The payment was processed.") >= 1

    def test_multiple_keywords_accumulate(self, model):
        sentence = "The fintech startup uses AI and blockchain for payments."
        score = model._score_sentence(sentence)
        assert score >= 3  # fintech, ai, blockchain, payment — at least 3

    def test_scoring_is_case_insensitive(self, model):
        lower = model._score_sentence("the payment system is growing.")
        upper = model._score_sentence("The PAYMENT system is growing.")
        assert lower == upper


# ─── _is_duplicate ────────────────────────────────────────────────────────────

class TestIsDuplicate:
    def test_identical_sentences_are_duplicate(self, model):
        sent = "Digital payments are transforming financial services."
        assert model._is_duplicate(sent, [sent]) is True

    def test_completely_different_sentences_not_duplicate(self, model):
        sent = "Blockchain enables decentralized finance globally."
        existing = ["The weather is sunny and warm outside today."]
        assert model._is_duplicate(sent, existing) is False

    def test_empty_selected_list_never_duplicate(self, model):
        assert model._is_duplicate("Any sentence here.", []) is False

    def test_high_overlap_above_threshold_is_duplicate(self, model):
        sent = "Fintech payments are growing fast in emerging markets."
        similar = "Fintech payments are growing fast across emerging markets."
        assert model._is_duplicate(sent, [similar]) is True

    def test_low_overlap_below_threshold_not_duplicate(self, model):
        sent = "AI is transforming the lending industry worldwide."
        different = "Blockchain enables decentralized peer-to-peer transactions."
        assert model._is_duplicate(sent, [different]) is False

    def test_custom_threshold_zero_always_duplicate(self, model):
        sent = "Payment systems are evolving rapidly."
        other = "Completely unrelated sentence about weather."
        # threshold=0 means any overlap → duplicate
        assert model._is_duplicate(sent, [other], threshold=0.0) is True


# ─── summarize ───────────────────────────────────────────────────────────────

class TestSummarize:
    def test_returns_string(self, model):
        docs = [_doc("Stripe raised $1 billion in funding. The company plans to expand globally into payments.")]
        result = model.summarize(docs)
        assert isinstance(result, str)

    def test_empty_documents_returns_fallback(self, model):
        result = model.summarize([_doc("")])
        assert result == "No content to summarize."

    def test_no_documents_returns_fallback(self, model):
        result = model.summarize([])
        assert result == "No content to summarize."

    def test_summarizes_single_document(self, model):
        text = (
            "Stripe raised $1 billion in Series H funding. "
            "The payment company plans to expand into Southeast Asia. "
            "Digital banking is becoming mainstream among millennials. "
            "AI is transforming the lending and credit scoring industry. "
            "Blockchain technology enables decentralized financial transactions."
        )
        result = model.summarize([_doc(text)])
        assert len(result) > 0
        assert result != "No content to summarize."

    def test_summarizes_multiple_documents(self, model):
        docs = [
            _doc("Stripe raised $1 billion for payment infrastructure expansion globally."),
            _doc("Neobanks are disrupting traditional banking with digital-first financial services."),
        ]
        result = model.summarize(docs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_does_not_exceed_top_7_sentences(self, model):
        """At most 7 sentences in output."""
        # 10 distinct complete sentences about fintech
        sentences = [
            "Stripe raised $1 billion in Series H funding for global expansion.",
            "The payment company targets Southeast Asian markets for growth.",
            "Neobanks are disrupting traditional banking with digital services.",
            "AI-powered credit scoring reduces loan default rates significantly.",
            "Blockchain enables transparent and secure cross-border payments.",
            "Embedded finance allows any app to offer banking as a service.",
            "RegTech solutions automate compliance monitoring for financial firms.",
            "Buy-now-pay-later lending is growing rapidly among young consumers.",
            "Robo-advisors offer automated portfolio management for retail investors.",
            "Crypto wallets are becoming mainstream for digital asset management.",
        ]
        docs = [_doc(" ".join(sentences))]
        result = model.summarize(docs)
        # Count by splitting on sentence-terminal punctuation
        output_sentences = [s for s in result.split(". ") if s.strip()]
        assert len(output_sentences) <= 7

    def test_high_scoring_sentences_are_preferred(self, model):
        """Sentences with more fintech keywords should appear in output."""
        text = (
            "The sky is blue and the weather is nice today outside. "  # 0 fintech keywords
            "AI-powered payment fintech platforms are revolutionizing digital banking globally."  # many keywords
        )
        result = model.summarize([_doc(text)])
        assert "fintech" in result.lower() or "payment" in result.lower() or "banking" in result.lower()
