"""Unit tests for service layer."""

import asyncio
from datetime import date
from unittest.mock import Mock

from langchain_core.documents import Document

from core.models.trend_metric import TrendMetric
from core.services.ingestion_service import article_to_document
from core.models.thesis import StructuredThesis
from core.services.thesis_generator_service import (
    ThesisGeneratorService,
    _apply_cap_deltas,
    _gold_confidence_inputs,
    _ranked_tags_from_documents,
    _select_feedback_evidence,
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

    def __init__(self, docs=None):
        self.retrieve_args = None
        self._docs = docs if docs is not None else [
            Document(page_content="r", metadata={"url": "u"})
        ]

    def build(self, documents):
        return object()

    def retrieve(
        self,
        query,
        fetch_k,
        max_articles,
        window_days=None,
        query_embedding=None,
        min_similarity=0.0,
        date_from=None,
        date_to=None,
    ):
        self.retrieve_args = {
            "query": query,
            "fetch_k": fetch_k,
            "max_articles": max_articles,
            "window_days": window_days,
            "query_embedding": query_embedding,
            "min_similarity": min_similarity,
            "date_from": date_from,
            "date_to": date_to,
        }
        return self._docs


class TestDocumentRetrievalService:
    """Tests for DocumentRetrievalService wiring.

    The service is stateless (no open/build step): retrieve() delegates the wide
    pool query straight to the vector store, and select_diverse() runs MMR over
    the result to pick the LLM's subset."""

    def _service(self, config, docs=None):
        from core.services.retrieval_service import DocumentRetrievalService

        vs = _RecordingVectorStore(docs=docs)
        return DocumentRetrievalService(vs, config), vs

    def test_retrieve_uses_pool_config(self):
        from config.settings import RetrievalConfig

        service, vs = self._service(
            RetrievalConfig(k=5, fetch_k=400, max_articles=50, window_days=365)
        )
        docs = service.retrieve("query")

        # The wide-pool sizing (chunk pool + article cap) must reach the store;
        # k/lambda_mult are the LLM-subset dials and belong to select_diverse.
        assert vs.retrieve_args["fetch_k"] == 400
        assert vs.retrieve_args["max_articles"] == 50
        # The configured relevance floor must reach the vector store.
        assert vs.retrieve_args["min_similarity"] == 0.72
        assert len(docs) == 1

    def test_retrieve_passes_window_days(self):
        from config.settings import RetrievalConfig

        # The configured recency window must reach the vector store.
        service, vs = self._service(RetrievalConfig(window_days=180))
        service.retrieve("query")

        assert vs.retrieve_args["window_days"] == 180

    def test_select_diverse_mmr_narrows_to_k(self):
        from config.settings import RetrievalConfig

        # Wide pool of 4 articles, each with an embedding; MMR must pick k=2.
        pool = [
            Document(page_content=f"a{i}", metadata={"url": f"u{i}", "embedding": vec})
            for i, vec in enumerate(([1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.5, 0.5]))
        ]
        service, _ = self._service(RetrievalConfig(k=2, lambda_mult=0.5), docs=pool)

        selected = service.select_diverse(pool, query_embedding=[1.0, 0.0])

        assert len(selected) == 2
        # The transient embedding must not survive into the LLM/persisted docs.
        assert all("embedding" not in d.metadata for d in selected)

    def test_select_diverse_skips_mmr_when_pool_at_or_below_k(self):
        from config.settings import RetrievalConfig

        # 3 docs, k=5 -> no selection needed; pass all through in order without
        # needing a query embedding at all, embedding stripped.
        pool = [
            Document(page_content=f"a{i}", metadata={"url": f"u{i}", "embedding": [1.0, 0.0]})
            for i in range(3)
        ]
        service, _ = self._service(RetrievalConfig(k=5), docs=pool)

        selected = service.select_diverse(pool, query_embedding=None)

        assert [d.metadata["url"] for d in selected] == ["u0", "u1", "u2"]
        assert all("embedding" not in d.metadata for d in selected)

    def test_select_diverse_falls_back_without_query_embedding(self):
        from config.settings import RetrievalConfig

        pool = [
            Document(page_content=f"a{i}", metadata={"url": f"u{i}", "embedding": [1.0, 0.0]})
            for i in range(4)
        ]
        service, _ = self._service(RetrievalConfig(k=2), docs=pool)

        # No query vector -> top-k by relevance order, embedding still stripped.
        selected = service.select_diverse(pool, query_embedding=None)

        assert [d.metadata["url"] for d in selected] == ["u0", "u1"]
        assert all("embedding" not in d.metadata for d in selected)

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


def _pool_doc(url, *, themes=None, signals=None, published="2026-01-01", sim=0.75):
    return Document(page_content=url, metadata={
        "url": url,
        "themes": themes or [],
        "signals": signals or [],
        "published_at": f"{published}T00:00:00",
        "similarity": sim,
    })


class TestSelectFeedbackEvidence:
    """_select_feedback_evidence: the refinement evidence lens. Deterministic
    re-ranking of the wide pool by stored metadata, blended with the original
    summary docs for continuity. Only touches which docs the LLM reads."""

    # A wide pool of distinct articles with varied tags / dates / similarity.
    POOL = [
        _pool_doc("a", themes=["Digital Payments"], signals=["Payment Infrastructure"], published="2026-01-01", sim=0.80),
        _pool_doc("b", themes=["Digital Lending"], published="2026-06-01", sim=0.78),
        _pool_doc("c", themes=["Digital Payments"], signals=["AI-Driven Financial Tools"], published="2026-07-01", sim=0.75),
        _pool_doc("d", themes=["WealthTech"], signals=["WealthTech Disruption"], published="2026-03-01", sim=0.90),
        _pool_doc("e", published="2026-02-01", sim=0.73),
    ]
    # The current summary subset (distinct from the pool urls) -> top 2 kept.
    ORIGINAL = [_pool_doc("o1"), _pool_doc("o2"), _pool_doc("o3")]

    def _urls(self, docs):
        return [d.metadata["url"] for d in docs]

    def _run(self, feedback, key_themes=("Digital Payments",)):
        thesis = StructuredThesis(key_themes=list(key_themes))
        return _select_feedback_evidence(self.POOL, self.ORIGINAL, feedback, thesis, target_n=3)

    def test_structural_feedback_no_lens_keeps_original(self):
        out = self._run(["Too many risks, not enough opportunities"])
        assert self._urls(out) == ["o1", "o2", "o3"]

    def test_theme_lens_blends_theme_tagged_by_similarity(self):
        # key theme "Digital Payments" -> a(0.80), c(0.75); keep top-2 original + a.
        out = self._run(["Need stronger evidence for key themes"])
        assert self._urls(out) == ["o1", "o2", "a"]

    def test_signal_lens_picks_signal_tagged_by_similarity(self):
        # signal-tagged: d(0.90), a(0.80), c(0.75) -> fill with d.
        out = self._run(["Investment signals are too vague"])
        assert self._urls(out) == ["o1", "o2", "d"]

    def test_recency_lens_picks_most_recent(self):
        # newest published: c (2026-07) -> fill with c.
        out = self._run(["Missing recent market trends"])
        assert self._urls(out) == ["o1", "o2", "c"]

    def test_focus_lens_picks_highest_similarity(self):
        # tightest cluster: d(0.90) -> fill with d.
        out = self._run(["Analysis is too broad, be more specific"])
        assert self._urls(out) == ["o1", "o2", "d"]

    def test_lens_with_no_candidates_falls_back_to_original(self):
        # theme lens but no pool article carries the key theme -> unchanged.
        out = self._run(["Need stronger evidence for key themes"], key_themes=("Insurtech",))
        assert self._urls(out) == ["o1", "o2", "o3"]


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

        # Tag strength must clear the deterministic gate-2 floor (see
        # MIN_*_STRENGTH_FOR_SUMMARY) or generate_thesis skips mock_llm.summarize
        # entirely and this test would pass without ever exercising it.
        docs = [Document(page_content="Digital banking is the future", metadata={
            "url": "http://test.com",
            "themes": ["Digital Banking"] * 3,
            "risks": ["Regulatory Risk"] * 2,
            "signals": ["Payment Infrastructure"] * 2,
        })]
        thesis = asyncio.run(service.generate_thesis("Digital Banking", docs))

        # Content reflects MockLanguageModel's actual return value - proves the
        # mock was called, not just that raw_output is non-empty.
        assert thesis.raw_output is not None
        assert thesis.raw_output.startswith("Mock summary:")
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

        async def local_summarize(documents, topic=""):
            summary_source_var.set(SOURCE_LOCAL)
            return "extractive summary"

        llm.summarize = local_summarize
        service = ThesisGeneratorService(llm, OpportunityScoringService(), _empty_trend())

        # Tag strength must clear the deterministic gate-2 floor (see
        # MIN_*_STRENGTH_FOR_SUMMARY) or generate_thesis skips the summarize
        # call entirely and never reaches this mock's local-fallback branch.
        docs = [Document(page_content="x", metadata={
            "url": "http://test.com",
            "themes": ["Digital Banking"] * 3,
            "risks": ["Regulatory Risk"] * 2,
            "signals": ["Payment Infrastructure"] * 2,
        })]
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
