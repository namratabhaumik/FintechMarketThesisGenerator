"""Unit tests for service layer."""

from langchain_core.documents import Document

from core.services.ingestion_service import ArticleIngestionService
from core.services.thesis_generator_service import (
    ThesisGeneratorService,
    _apply_feedback_caps,
    _ranked_tags_from_documents,
)


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
        assert "published_at" in doc.metadata
        assert doc.metadata["title"] == "Test Article 1"
        assert doc.metadata["published_at"] == "2026-01-01T00:00:00+00:00"


class _RecordingVectorStore:
    """Fake IVectorStore that records the args retrieve() was called with."""

    def __init__(self):
        self.retrieve_args = None

    def build(self, documents):
        return object()

    def open(self):
        return object()

    def retrieve(self, vectorstore, query, k, fetch_k, lambda_mult, window_days=None):
        self.retrieve_args = {
            "query": query,
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "window_days": window_days,
        }
        return [Document(page_content="r", metadata={"url": "u"})]


class TestDocumentRetrievalService:
    """Tests for DocumentRetrievalService MMR wiring."""

    def _service(self, config):
        from core.services.retrieval_service import DocumentRetrievalService

        vs = _RecordingVectorStore()
        service = DocumentRetrievalService(vs, config)
        service.build_vectorstore([Document(page_content="x", metadata={"url": "u"})])
        return service, vs

    def test_is_built_false_initially(self):
        from config.settings import RetrievalConfig
        from core.services.retrieval_service import DocumentRetrievalService

        service = DocumentRetrievalService(_RecordingVectorStore(), RetrievalConfig())
        assert service.is_built() is False

    def test_retrieve_lazily_opens_existing_store(self):
        from config.settings import RetrievalConfig
        from core.services.retrieval_service import DocumentRetrievalService

        # No build_vectorstore() call: retrieve must open the existing store
        vs = _RecordingVectorStore()
        service = DocumentRetrievalService(vs, RetrievalConfig())
        assert service.is_built() is False

        docs = service.retrieve("query")

        assert service.is_built() is True
        assert len(docs) == 1

    def test_retrieve_uses_mmr_config(self):
        from config.settings import RetrievalConfig

        service, vs = self._service(
            RetrievalConfig(k=5, fetch_k=20, lambda_mult=0.5, window_days=365)
        )
        docs = service.retrieve("query")

        assert vs.retrieve_args["k"] == 5
        assert vs.retrieve_args["fetch_k"] == 20
        assert vs.retrieve_args["lambda_mult"] == 0.5
        assert len(docs) == 1

    def test_retrieve_passes_window_days(self):
        from config.settings import RetrievalConfig

        # The configured recency window must reach the vector store.
        service, vs = self._service(RetrievalConfig(window_days=180))
        service.retrieve("query")

        assert vs.retrieve_args["window_days"] == 180

    def test_retrieve_override_k_widens_fetch_k(self):
        from config.settings import RetrievalConfig

        # k override above the configured fetch_k must widen fetch_k (MMR needs
        # fetch_k >= k).
        service, vs = self._service(RetrievalConfig(k=5, fetch_k=20, lambda_mult=0.5))
        service.retrieve("query", k=30)

        assert vs.retrieve_args["k"] == 30
        assert vs.retrieve_args["fetch_k"] == 30


class TestThesisGeneratorService:
    """Tests for ThesisGeneratorService."""

    def test_generate_thesis_structure(self, mock_llm, mock_scoring_strategy):
        """Test that thesis has correct structure."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service)

        docs = [Document(page_content="Test content", metadata={"url": "http://test.com"})]
        thesis = service.generate_thesis("Digital Banking", docs)

        assert thesis.key_themes is not None
        assert thesis.risks is not None
        assert thesis.investment_signals is not None
        assert thesis.sources is not None

    def test_generate_thesis_with_mock_llm(self, mock_llm, mock_scoring_strategy):
        """Test thesis generation with mock LLM."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service)

        docs = [Document(page_content="Digital banking is the future", metadata={"url": "http://test.com"})]
        thesis = service.generate_thesis("Digital Banking", docs)

        assert thesis.raw_output is not None
        assert thesis.sources == ["http://test.com"]

    def test_generate_thesis_includes_opportunity_score(self, mock_llm, mock_scoring_strategy):
        """Test that thesis includes opportunity score and confidence."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service)

        docs = [
            Document(page_content="Digital banking innovation", metadata={"url": "http://test.com"}),
            Document(page_content="Payment systems growth", metadata={"url": "http://test2.com"}),
        ]
        thesis = service.generate_thesis("Digital Banking", docs)

        assert hasattr(thesis, "opportunity_score")
        assert hasattr(thesis, "confidence_level")
        assert hasattr(thesis, "recommendation")
        assert hasattr(thesis, "key_risk_factors")
        assert 0 <= thesis.opportunity_score <= 5.0
        assert 0 <= thesis.confidence_level <= 1.0
        assert thesis.recommendation in ["Pursue", "Investigate", "Skip"]

    def test_generate_thesis_recommendation_based_on_score(self, mock_llm, mock_scoring_strategy):
        """Test that recommendation matches score thresholds."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service)

        docs = [Document(page_content="Test", metadata={"url": "http://test.com"})]
        thesis = service.generate_thesis("Test Query", docs)

        # Verify recommendation is consistent with score
        if thesis.opportunity_score >= 3.75:
            assert thesis.recommendation == "Pursue"
        elif thesis.opportunity_score >= 2.5:
            assert thesis.recommendation == "Investigate"
        else:
            assert thesis.recommendation == "Skip"


class TestGroundedTagDerivation:
    """Thesis tags come only from the retrieved docs' Silver metadata; feedback
    adjusts how many surface but never invents tags absent from the evidence."""

    @staticmethod
    def _doc(themes=None, risks=None, signals=None, url="u"):
        return Document(
            page_content="x",
            metadata={
                "url": url,
                "themes": themes or [],
                "risks": risks or [],
                "signals": signals or [],
            },
        )

    def test_ranked_tags_are_frequency_ordered(self):
        docs = [
            self._doc(themes=["Payments", "Crypto"]),
            self._doc(themes=["Payments"]),
            self._doc(themes=["Payments", "Lending"]),
        ]
        themes, risks, signals = _ranked_tags_from_documents(docs)
        assert themes[0] == "Payments"  # most frequent first
        assert set(themes) == {"Payments", "Crypto", "Lending"}
        assert risks == [] and signals == []

    def test_missing_metadata_treated_as_no_tags(self):
        docs = [Document(page_content="x", metadata={"url": "u"})]  # no tag keys
        assert _ranked_tags_from_documents(docs) == ([], [], [])

    def test_feedback_trims_risks_and_expands_signals(self):
        kt, kr, ks = _apply_feedback_caps(
            ["T1", "T2", "T3", "T4"],
            ["R1", "R2", "R3", "R4"],
            ["S1", "S2", "S3", "S4"],
            ["Too many risks, not enough opportunities"],
            base_cap=3,
        )
        assert kr == ["R1", "R2"]                 # risks trimmed to base-1
        assert ks == ["S1", "S2", "S3", "S4"]     # signals expanded to base+1
        assert kt == ["T1", "T2", "T3"]           # themes unchanged at base

    def test_feedback_expansion_bounded_by_evidence(self):
        # "score too low" expands signals, but only as far as the evidence goes.
        _, _, ks = _apply_feedback_caps(
            ["T1"], ["R1"], ["S1", "S2"],
            ["Opportunity score seems too low"],
            base_cap=3,
        )
        assert ks == ["S1", "S2"]  # base+1=4 requested, only 2 grounded signals

    def test_narrative_only_feedback_leaves_caps_at_base(self):
        result = _apply_feedback_caps(
            ["T1", "T2", "T3", "T4"],
            ["R1", "R2", "R3", "R4"],
            ["S1", "S2", "S3", "S4"],
            ["Need stronger evidence for key themes"],
            base_cap=3,
        )
        assert result == (["T1", "T2", "T3"], ["R1", "R2", "R3"], ["S1", "S2", "S3"])

    def test_generate_thesis_surfaces_grounded_tags(self, mock_llm):
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        service = ThesisGeneratorService(mock_llm, OpportunityScoringService())
        docs = [
            self._doc(
                themes=["Digital Payments"],
                risks=["Regulatory Risk"],
                signals=["Payment Infrastructure"],
                url="u1",
            ),
            self._doc(themes=["Digital Payments"], url="u2"),
        ]
        thesis = service.generate_thesis("payments", docs)

        assert thesis.key_themes == ["Digital Payments"]
        assert thesis.risks == ["Regulatory Risk"]
        assert thesis.investment_signals == ["Payment Infrastructure"]
        assert thesis.sources == ["u1", "u2"]
