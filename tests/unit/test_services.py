"""Unit tests for service layer."""

import pytest
from langchain.docstore.document import Document

from core.services.ingestion_service import ArticleIngestionService
from core.services.retrieval_service import DocumentRetrievalService
from core.services.thesis_generator_service import ThesisGeneratorService


class TestArticleIngestionService:
    """Tests for ArticleIngestionService."""

    def test_fetch_articles(self, mock_article_source):
        """Test fetching articles."""
        service = ArticleIngestionService(mock_article_source)
        articles = service.fetch_articles("fintech", limit=5)

        assert len(articles) > 0
        assert articles[0].title == "Test Article 1"
        assert articles[0].source == "example.com"

    def test_fetch_articles_respects_limit(self, mock_article_source):
        """Test that fetch respects limit."""
        service = ArticleIngestionService(mock_article_source)
        articles = service.fetch_articles("fintech", limit=2)

        assert len(articles) == 2

    def test_convert_to_documents(self, sample_articles, mock_article_source):
        """Test conversion of articles to documents."""
        service = ArticleIngestionService(mock_article_source)
        docs = service.convert_to_documents(sample_articles)

        assert len(docs) == 3
        assert isinstance(docs[0], Document)
        assert "Test Article 1" in docs[0].page_content
        assert docs[0].metadata["source"] == "example.com"

    def test_convert_to_documents_includes_metadata(self, sample_articles, mock_article_source):
        """Test that documents include proper metadata."""
        service = ArticleIngestionService(mock_article_source)
        docs = service.convert_to_documents(sample_articles)

        doc = docs[0]
        assert "title" in doc.metadata
        assert "source" in doc.metadata
        assert "url" in doc.metadata
        assert doc.metadata["title"] == "Test Article 1"


class TestDocumentRetrievalService:
    """Tests for DocumentRetrievalService."""

    def test_build_vectorstore(self, mock_llm):
        """Test that vectorstore is built from documents."""
        from tests.conftest import MockLanguageModel
        from core.implementations.embeddings.huggingface_embeddings import HuggingFaceEmbeddingModel
        from core.implementations.vectorstores.faiss_store import FAISSVectorStore
        from config.settings import EmbeddingConfig, VectorStoreConfig

        # This would require actual HuggingFace embeddings, so we skip in test
        # In production, use mocks or skip this test
        pytest.skip("Requires HuggingFace embeddings initialization")

    def test_is_built_false_initially(self, mock_llm):
        """Test that vectorstore is not built initially."""
        from core.implementations.vectorstores.faiss_store import FAISSVectorStore
        from config.settings import VectorStoreConfig
        from core.implementations.embeddings.huggingface_embeddings import HuggingFaceEmbeddingModel
        from config.settings import EmbeddingConfig

        # Create minimal setup to test is_built() without actual vectorstore
        pytest.skip("Requires full setup")


class TestThesisGeneratorService:
    """Tests for ThesisGeneratorService."""

    def test_generate_thesis_structure(self, mock_llm, mock_scoring_strategy):
        """Test that thesis has correct structure."""
        from core.services.thesis_structuring_service import ThesisStructuringService

        structurer = ThesisStructuringService(mock_scoring_strategy)
        service = ThesisGeneratorService(mock_llm, structurer)

        docs = [Document(page_content="Test content", metadata={"url": "http://test.com"})]
        thesis = service.generate_thesis("Digital Banking", docs)

        assert thesis.key_themes is not None
        assert thesis.risks is not None
        assert thesis.investment_signals is not None
        assert thesis.sources is not None

    def test_generate_thesis_with_mock_llm(self, mock_llm, mock_scoring_strategy):
        """Test thesis generation with mock LLM."""
        from core.services.thesis_structuring_service import ThesisStructuringService

        structurer = ThesisStructuringService(mock_scoring_strategy)
        service = ThesisGeneratorService(mock_llm, structurer)

        docs = [Document(page_content="Digital banking is the future", metadata={"url": "http://test.com"})]
        thesis = service.generate_thesis("Digital Banking", docs)

        assert thesis.raw_output is not None
        assert thesis.sources == ["http://test.com"]
