"""Unit tests for service layer."""

import asyncio
from datetime import date
from unittest.mock import Mock

from langchain_core.documents import Document

from core.models.trend_metric import TrendMetric
from core.services.ingestion_service import article_to_document
from core.services.thesis_generator_service import (
    ThesisGeneratorService,
    _apply_cap_deltas,
    _gold_confidence_inputs,
    _ranked_tags_from_documents,
)


def _empty_trend():
    """Trend repo stub with no Gold metrics (confidence falls to 0)."""
    repo = Mock()
    repo.fetch_all.return_value = []
    repo.fetch_recent.return_value = []
    return repo


class TestGoldConfidenceInputs:
    """_gold_confidence_inputs: covered weeks for the evidence's categories over
    the window, derived from Gold; window is the retrieval window in weeks (or the
    full Gold span when None)."""

    # Four consecutive Mondays (the recent window) plus one older Monday.
    W0, W1, W2, W3 = (
        date(2026, 6, 15), date(2026, 6, 8), date(2026, 6, 1), date(2026, 5, 25),
    )
    OLD = date(2026, 5, 4)

    @staticmethod
    def _doc():
        return Document(page_content="x", metadata={
            "url": "u", "themes": ["Payments"], "risks": ["Reg"], "signals": ["Infra"],
        })

    def _metrics(self):
        return [
            TrendMetric(week_start=self.W0, dimension="theme", category="Payments", article_count=5),
            TrendMetric(week_start=self.W2, dimension="theme", category="Payments", article_count=3),
            TrendMetric(week_start=self.W1, dimension="signal", category="Infra", article_count=2),
            TrendMetric(week_start=self.W0, dimension="theme", category="Other", article_count=9),
            TrendMetric(week_start=self.OLD, dimension="theme", category="Payments", article_count=4),
        ]

    def test_empty_gold_gives_zero_over_window(self):
        assert _gold_confidence_inputs([self._doc()], [], 4) == (0, 4, None)
        # window None (whole corpus) with no metrics still avoids a zero denominator.
        assert _gold_confidence_inputs([self._doc()], [], None) == (0, 1, None)

    def test_windowed_counts_matching_weeks_in_window(self):
        # Matching weeks: W0, W2 (Payments), W1 (Infra), OLD (Payments, outside window).
        # "Other" is not in the evidence -> ignored. Window = {W0, W1, W2, W3}.
        covered, window_weeks, as_of = _gold_confidence_inputs(
            [self._doc()], self._metrics(), 4
        )
        assert (covered, window_weeks, as_of) == (3, 4, self.W0)

    def test_whole_corpus_spans_full_gold(self):
        # window None -> denominator is the full span (W0..OLD = 7 weeks inclusive),
        # covered = all distinct matching weeks (W0, W1, W2, OLD).
        covered, window_weeks, as_of = _gold_confidence_inputs(
            [self._doc()], self._metrics(), None
        )
        assert (covered, window_weeks, as_of) == (4, 7, self.W0)


class TestArticleToDocument:
    """article_to_document: the shared Article -> Document conversion."""

    def test_converts_articles_to_documents(self, sample_articles):
        docs = [article_to_document(a) for a in sample_articles]

        assert len(docs) == 3
        assert isinstance(docs[0], Document)
        assert "Test Article 1" in docs[0].page_content
        assert docs[0].metadata["source"] == "example.com"

    def test_documents_include_metadata(self, sample_articles):
        doc = article_to_document(sample_articles[0])

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

    def retrieve(
        self,
        query,
        k,
        fetch_k,
        lambda_mult,
        window_days=None,
        query_embedding=None,
        min_similarity=0.0,
        date_from=None,
        date_to=None,
    ):
        self.retrieve_args = {
            "query": query,
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "window_days": window_days,
            "query_embedding": query_embedding,
            "min_similarity": min_similarity,
            "date_from": date_from,
            "date_to": date_to,
        }
        return [Document(page_content="r", metadata={"url": "u"})]


class TestDocumentRetrievalService:
    """Tests for DocumentRetrievalService MMR wiring.

    The service is stateless (no open/build step): retrieve() delegates
    straight to the vector store implementation."""

    def _service(self, config):
        from core.services.retrieval_service import DocumentRetrievalService

        vs = _RecordingVectorStore()
        return DocumentRetrievalService(vs, config), vs

    def test_retrieve_uses_mmr_config(self):
        from config.settings import RetrievalConfig

        service, vs = self._service(
            RetrievalConfig(k=5, fetch_k=20, lambda_mult=0.5, window_days=365)
        )
        docs = service.retrieve("query")

        assert vs.retrieve_args["k"] == 5
        assert vs.retrieve_args["fetch_k"] == 20
        assert vs.retrieve_args["lambda_mult"] == 0.5
        # The configured relevance floor must reach the vector store.
        assert vs.retrieve_args["min_similarity"] == 0.72
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

    def test_retrieve_query_date_intent_overrides_window_days(self):
        from config.settings import RetrievalConfig

        # A query naming an explicit date range takes over from the default
        # trailing window instead of being combined with it.
        service, vs = self._service(RetrievalConfig(window_days=365))
        service.retrieve("fintech regulation since March 2024")

        assert vs.retrieve_args["window_days"] is None
        assert vs.retrieve_args["date_from"] is not None
        assert vs.retrieve_args["date_to"] is not None

    def test_retrieve_query_without_date_intent_keeps_window_days(self):
        from config.settings import RetrievalConfig

        service, vs = self._service(RetrievalConfig(window_days=365))
        service.retrieve("crypto adoption in Asia")

        assert vs.retrieve_args["window_days"] == 365
        assert vs.retrieve_args["date_from"] is None
        assert vs.retrieve_args["date_to"] is None


class TestThesisGeneratorService:
    """Tests for ThesisGeneratorService."""

    def test_generate_thesis_structure(self, mock_llm, mock_scoring_strategy):
        """Test that thesis has correct structure."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service, _empty_trend())

        docs = [Document(page_content="Test content", metadata={"url": "http://test.com"})]
        thesis = asyncio.run(service.generate_thesis("Digital Banking", docs))

        assert thesis.key_themes is not None
        assert thesis.risks is not None
        assert thesis.investment_signals is not None
        assert thesis.sources is not None

    def test_generate_thesis_with_mock_llm(self, mock_llm, mock_scoring_strategy):
        """Test thesis generation with mock LLM."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service, _empty_trend())

        docs = [Document(page_content="Digital banking is the future", metadata={"url": "http://test.com"})]
        thesis = asyncio.run(service.generate_thesis("Digital Banking", docs))

        assert thesis.raw_output is not None
        assert thesis.sources == ["http://test.com"]
        # A plain LLM summary carries the default provenance.
        assert thesis.summary_source == "llm"

    def test_generate_thesis_records_local_fallback_provenance(self):
        """When the local extractive path produces the summary (it sets the
        provenance var, like the real LocalSummarizerModel), the thesis
        records summary_source='local'."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        from core.interfaces.llm import SOURCE_LOCAL, summary_source_var

        llm = Mock()

        async def local_summarize(documents):
            summary_source_var.set(SOURCE_LOCAL)
            return "extractive summary"

        llm.summarize = local_summarize
        service = ThesisGeneratorService(llm, OpportunityScoringService(), _empty_trend())

        docs = [Document(page_content="x", metadata={"url": "http://test.com"})]
        thesis = asyncio.run(service.generate_thesis("Digital Banking", docs))

        assert thesis.summary_source == "local"

    def test_generate_thesis_includes_opportunity_score(self, mock_llm, mock_scoring_strategy):
        """Test that thesis includes opportunity score and confidence."""
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        scoring_service = OpportunityScoringService()
        service = ThesisGeneratorService(mock_llm, scoring_service, _empty_trend())

        docs = [
            Document(page_content="Digital banking innovation", metadata={"url": "http://test.com"}),
            Document(page_content="Payment systems growth", metadata={"url": "http://test2.com"}),
        ]
        thesis = asyncio.run(service.generate_thesis("Digital Banking", docs))

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
        service = ThesisGeneratorService(mock_llm, scoring_service, _empty_trend())

        docs = [Document(page_content="Test", metadata={"url": "http://test.com"})]
        thesis = asyncio.run(service.generate_thesis("Test Query", docs))

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

    def test_planner_deltas_trim_risks_and_expand_signals(self):
        kt, kr, ks = _apply_cap_deltas(
            ["T1", "T2", "T3", "T4"],
            ["R1", "R2", "R3", "R4"],
            ["S1", "S2", "S3", "S4"],
            theme_delta=0, risk_delta=-1, signal_delta=1,
            base_cap=3,
        )
        assert kr == ["R1", "R2"]                 # risks trimmed to base-1
        assert ks == ["S1", "S2", "S3", "S4"]     # signals expanded to base+1
        assert kt == ["T1", "T2", "T3"]           # themes unchanged at base

    def test_delta_expansion_bounded_by_evidence(self):
        # +1 expands signals, but only as far as the evidence goes.
        _, _, ks = _apply_cap_deltas(
            ["T1"], ["R1"], ["S1", "S2"],
            theme_delta=0, risk_delta=0, signal_delta=1,
            base_cap=3,
        )
        assert ks == ["S1", "S2"]  # base+1=4 requested, only 2 grounded signals

    def test_zero_deltas_leave_caps_at_base(self):
        result = _apply_cap_deltas(
            ["T1", "T2", "T3", "T4"],
            ["R1", "R2", "R3", "R4"],
            ["S1", "S2", "S3", "S4"],
            theme_delta=0, risk_delta=0, signal_delta=0,
            base_cap=3,
        )
        assert result == (["T1", "T2", "T3"], ["R1", "R2", "R3"], ["S1", "S2", "S3"])

    def test_deltas_clamped_beyond_one(self):
        # Even if the LLM somehow returns an out-of-range delta, it's clamped to +-1.
        kt, kr, ks = _apply_cap_deltas(
            ["T1", "T2", "T3", "T4", "T5"],
            ["R1", "R2", "R3", "R4"],
            ["S1", "S2", "S3", "S4"],
            theme_delta=3, risk_delta=-5, signal_delta=0,
            base_cap=3,
        )
        assert kt == ["T1", "T2", "T3", "T4"]  # clamped to base+1, not base+3
        assert kr == ["R1", "R2"]              # clamped to base-1 (floor 1), not base-5

    def test_generate_thesis_surfaces_grounded_tags(self, mock_llm):
        from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

        service = ThesisGeneratorService(mock_llm, OpportunityScoringService(), _empty_trend())
        docs = [
            self._doc(
                themes=["Digital Payments"],
                risks=["Regulatory Risk"],
                signals=["Payment Infrastructure"],
                url="u1",
            ),
            self._doc(themes=["Digital Payments"], url="u2"),
        ]
        thesis = asyncio.run(service.generate_thesis("payments", docs))

        assert thesis.key_themes == ["Digital Payments"]
        assert thesis.risks == ["Regulatory Risk"]
        assert thesis.investment_signals == ["Payment Infrastructure"]
        assert thesis.sources == ["u1", "u2"]
