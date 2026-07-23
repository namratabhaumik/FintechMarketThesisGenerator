"""Thesis generator service."""

import asyncio
import logging
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import FEEDBACK_LENS
from core.interfaces.llm import SOURCE_LLM, ILanguageModel, summary_source_var
from core.interfaces.trend_repository import ITrendRepository
from core.models.thesis import StructuredThesis
from core.models.trend_metric import TrendMetric
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

logger = logging.getLogger(__name__)


def _tag_counters(documents: List[Document]) -> Dict[str, "Counter[str]"]:
    """Count Silver-tag occurrences per dimension across the retrieved chunks.

    Each chunk was tagged deterministically on its raw article text at Silver
    time. Missing metadata (e.g. older chunks embedded before tagging existed) is
    treated as no tags.
    """
    counters: Dict[str, "Counter[str]"] = {
        "themes": Counter(),
        "risks": Counter(),
        "signals": Counter(),
    }
    for doc in documents:
        metadata = doc.metadata or {}
        for key, counter in counters.items():
            # `or []` guards both an absent key and an explicit None.
            counter.update(metadata.get(key) or [])
    return counters


def _ranked_tags_from_documents(
    documents: List[Document],
) -> Tuple[List[str], List[str], List[str]]:
    """Frequency-rank the Silver tags carried in the retrieved chunks' metadata.

    Returns the full ranked candidate list per dimension (themes, risks,
    signals), most common first and UNCAPPED. These are the only tags the thesis
    can ever surface - they come straight from the evidence, never invented.
    """
    counters = _tag_counters(documents)

    def _ranked(counter: "Counter[str]") -> List[str]:
        return [label for label, _ in counter.most_common()]

    return (
        _ranked(counters["themes"]),
        _ranked(counters["risks"]),
        _ranked(counters["signals"]),
    )


def _tag_strengths_from_documents(
    documents: List[Document],
) -> Tuple[int, int, int]:
    """Total Silver-tag occurrences per dimension across the retrieved chunks.

    Returns (signal_strength, theme_strength, risk_strength) - the grounded
    analog of the old keyword-hit counts, now summed from the Silver tags rather
    than scanned from the LLM prose. Computed from the FULL tags (uncapped), so
    the score is independent of how many tags are displayed and is identical for
    the same retrieved set - which is what keeps it stable across refinement.
    """
    counters = _tag_counters(documents)
    return (
        sum(counters["signals"].values()),
        sum(counters["themes"].values()),
        sum(counters["risks"].values()),
    )


def _gold_confidence_inputs(
    documents: List[Document],
    metrics: List[TrendMetric],
    window_weeks: Optional[int],
) -> Tuple[int, int, Optional[date]]:
    """Derive the Gold-based confidence inputs for the thesis's categories.

    Matches the FULL set of (dimension, category) tags the retrieved evidence
    carries - across all three dimensions - against the Gold trend metrics, and
    returns (covered_weeks, window_weeks, as_of):
      - covered_weeks: how many of the evidence window's (i.e. Gold weeks ending at
        the latest week) had any coverage of those categories.
      - window_weeks: the window size in weeks - the confidence denominator. When
        the caller passes None (whole-corpus retrieval) it is the full span of
        Gold, earliest week to latest, inclusive.
      - as_of: the latest week present in Gold (data freshness), or None if empty.

    The window is derived from the retrieval window, so a 1-year retrieval gives a
    52-week window, a 6-month retrieval a 26-week window, and so on. Uses the
    uncapped evidence tags, so the result is stable across a refinement session.
    """
    counters = _tag_counters(documents)
    evidence = (
        {("theme", c) for c in counters["themes"]}
        | {("risk", c) for c in counters["risks"]}
        | {("signal", c) for c in counters["signals"]}
    )
    if not metrics:
        return 0, (window_weeks or 1), None

    weeks_present = [m.week_start for m in metrics]
    as_of = max(weeks_present)
    matching_weeks = {m.week_start for m in metrics if (m.dimension, m.category) in evidence}

    if window_weeks is None:
        # Whole-corpus retrieval: the window spans all of Gold.
        window_weeks = (as_of - min(weeks_present)).days // 7 + 1
        covered_weeks = len(matching_weeks)
    else:
        window = {as_of - timedelta(weeks=i) for i in range(window_weeks)}
        covered_weeks = len(matching_weeks & window)

    return covered_weeks, max(1, window_weeks), as_of


def _apply_cap_deltas(
    themes: List[str],
    risks: List[str],
    signals: List[str],
    theme_delta: int,
    risk_delta: int,
    signal_delta: int,
    base_cap: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Apply the QUANTITATIVE effect of feedback to the grounded tags.

    Every feedback item also drives the LLM narrative rewrite (the qualitative
    effect, handled by llm.refine). On top of that, the planner LLM reads the
    same feedback and proposes a per-dimension delta on how many grounded tags
    to surface: +1 sharpens by surfacing one more concrete grounded tag (bounded
    by what the evidence has), -1 trims to the best-supported, 0 leaves the cap
    at base. Deltas are clamped to [-1, +1] here regardless of what the LLM
    returns.
    """
    def _clamp_cap(delta: int) -> int:
        return max(1, base_cap + max(-1, min(1, delta)))

    theme_cap = _clamp_cap(theme_delta)
    risk_cap = _clamp_cap(risk_delta)
    signal_cap = _clamp_cap(signal_delta)

    return themes[:theme_cap], risks[:risk_cap], signals[:signal_cap]


# How many of the original summary docs to keep when a lens swaps in fresh
# evidence, so the rewrite has continuity with the previous round instead of
# lurching to an entirely new set.
CONTINUITY_KEEP = 2


def _doc_url(doc: Document) -> str:
    return (doc.metadata or {}).get("url", "") or ""


def _doc_similarity(doc: Document) -> float:
    try:
        return float((doc.metadata or {}).get("similarity") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _lens_for_feedback(feedback_items: List[str]) -> Optional[str]:
    """The evidence lens the latest round asks for, or None.

    Maps this round's fixed feedback reasons to a lens via FEEDBACK_LENS. If a
    round selects several reasons, the first evidence-seeking one (in the user's
    selection order) wins; a round of only structural reasons returns None.
    """
    for item in feedback_items:
        lens = FEEDBACK_LENS.get(item)
        if lens:
            return lens
    return None


def _lens_candidates(
    lens: str, pool: List[Document], current_thesis: StructuredThesis
) -> List[Document]:
    """Rank the wide pool for the requested lens - deterministic, stored metadata
    only (Silver tags / published_at / query similarity), no embeddings."""
    if lens == "recency":
        # Most recently published first; published_at is an ISO string, so a
        # plain string sort is chronological.
        return sorted(pool, key=lambda d: (d.metadata or {}).get("published_at") or "", reverse=True)
    if lens == "focus":
        # Tightest, most query-relevant cluster - narrow a too-broad analysis.
        return sorted(pool, key=_doc_similarity, reverse=True)
    if lens == "theme":
        wanted = set(current_thesis.key_themes or [])
        tagged = [d for d in pool if wanted & set((d.metadata or {}).get("themes") or [])]
        return sorted(tagged, key=_doc_similarity, reverse=True)
    if lens == "signal":
        tagged = [d for d in pool if (d.metadata or {}).get("signals")]
        return sorted(tagged, key=_doc_similarity, reverse=True)
    return []


def _select_feedback_evidence(
    pool: List[Document],
    original_summary_docs: List[Document],
    feedback_items: List[str],
    current_thesis: StructuredThesis,
    target_n: int,
) -> List[Document]:
    """Pick the evidence the rewrite LLM reads, biased by the latest feedback.

    For an evidence-seeking round, blends continuity with fresh material: keeps
    the top CONTINUITY_KEEP of the original summary docs so the narrative does not
    lurch, then fills the rest with the highest-ranked lens candidates not already
    chosen (deduped by URL), up to `target_n`; if the lens ran short, tops up from
    the remaining original docs. For a structural round (no lens), or when the
    lens yields nothing, returns the original subset unchanged.

    Only affects which docs the LLM reads - tags and the frozen score come from
    the wide pool and are untouched.
    """
    lens = _lens_for_feedback(feedback_items)
    if lens is None:
        return original_summary_docs
    candidates = _lens_candidates(lens, pool, current_thesis)
    if not candidates:
        return original_summary_docs

    keep = original_summary_docs[:CONTINUITY_KEEP]
    seen = {_doc_url(d) for d in keep if _doc_url(d)}
    blended = list(keep)
    # Fresh lens-picked evidence, then a top-up from the remaining originals so we
    # always reach target_n when enough distinct docs exist.
    for source in (candidates, original_summary_docs):
        for d in source:
            if len(blended) >= target_n:
                break
            url = _doc_url(d)
            if url and url in seen:
                continue
            seen.add(url)
            blended.append(d)
    return blended[:target_n] if target_n else blended


class ThesisGeneratorService:
    """Service for generating market theses.

    Single Responsibility: Thesis generation orchestration. The LLM writes the
    narrative; the structured themes/risks/signals are derived deterministically
    from the retrieved docs' Silver tags
    """

    # Deterministic pre-LLM refusal floor for the summarize step: stricter than
    # the >0-per-dimension guard in api/routes.py's insufficient_evidence check
    # (which already ran before generate_thesis is called), so this only bites
    # when a dimension barely cleared that guard on a single incidental tag hit.
    MIN_THEME_STRENGTH_FOR_SUMMARY = 3
    MIN_RISK_STRENGTH_FOR_SUMMARY = 2
    MIN_SIGNAL_STRENGTH_FOR_SUMMARY = 2

    def __init__(
        self,
        llm: ILanguageModel,
        scoring_service: OpportunityScoringService,
        trend_repository: ITrendRepository,
        retrieval_window_days: int = 365,
        max_tags_per_dimension: int = 3,
    ):
        """Initialize with dependencies.

        Args:
            llm: Injected LLM implementation (for summarization).
            scoring_service: Injected opportunity scoring service.
            trend_repository: Gold-layer trend store, read to ground confidence in
                corpus coverage over the evidence window.
            retrieval_window_days: The retrieval recency window (RETRIEVAL_WINDOW_DAYS).
                The confidence window is derived from it - a 1-year retrieval gives
                a 52-week confidence window. 0 (whole corpus) -> full Gold span.
            max_tags_per_dimension: How many of the most common themes / risks /
                signals from the retrieved docs to surface per dimension.
        """
        self._llm = llm
        self._scoring_service = scoring_service
        self._trend_repository = trend_repository
        self._retrieval_window_days = retrieval_window_days
        self._max_tags = max_tags_per_dimension

    async def generate_thesis(
        self,
        topic: str,
        documents: List[Document],
        summary_documents: Optional[List[Document]] = None,
    ) -> StructuredThesis:
        """Generate structured thesis from documents.

        Args:
            topic: Market topic for analysis.
            documents: The wide analytics pool - the distinct articles retrieval
                surfaced. Tags, strengths, scoring, confidence and the sources
                list are all computed over this full set, so the numbers carry
                the real weight of the market.
            summary_documents: The diverse subset (MMR-selected) the LLM actually
                reads to write the narrative. Keeping this small holds token cost
                and latency flat while analytics stay wide. Defaults to
                `documents` when omitted (LLM reads the full pool).

        Returns:
            StructuredThesis object with analysis results.
        """
        logger.info(f"Generating thesis for topic: {topic}")
        summary_documents = summary_documents if summary_documents is not None else documents

        # Step 1: Tag strengths for scoring come from the WIDE pool (Step 5), so
        # score/confidence reflect true coverage. The LLM refusal FLOOR (Step 2)
        # is a separate check on the few docs the model will actually read, so
        # its thresholds stay calibrated to a small set.
        signal_strength, theme_strength, risk_strength = _tag_strengths_from_documents(
            documents
        )
        summary_signal, summary_theme, summary_risk = _tag_strengths_from_documents(
            summary_documents
        )

        # Step 2: Summarize documents. This is the ONLY LLM call - it writes the
        # narrative prose (raw_output); it does not decide the structured tags.
        # Below the tag-strength floor (measured on the LLM's own input docs),
        # the evidence the model would read is too thin for this specific query
        # to trust it to write a grounded narrative, so skip the call entirely
        # (see summary_status="refused" for the fallback path where the model
        # itself declines despite clearing this floor).
        if (
            summary_theme < self.MIN_THEME_STRENGTH_FOR_SUMMARY
            or summary_risk < self.MIN_RISK_STRENGTH_FOR_SUMMARY
            or summary_signal < self.MIN_SIGNAL_STRENGTH_FOR_SUMMARY
        ):
            logger.info(
                "Step 2: Tag strength below floor on summary docs "
                f"(theme={summary_theme}, risk={summary_risk}, signal={summary_signal}); "
                "skipping LLM summarize call"
            )
            summary = "REFUSED: "
            summary_status = "refused"
            summary_source = SOURCE_LLM
            refusal_reason: Optional[str] = "tag_strength_floor"
        else:
            # Reset provenance first: if the local extractive fallback ends up
            # producing the text (outage, cost limit, routing, cached local
            # response), it flips this to "local" and the thesis records it.
            logger.info("Step 2: Summarizing retrieved documents...")
            summary_source_var.set(SOURCE_LLM)
            summary = await self._llm.summarize(summary_documents, topic)
            summary_source = summary_source_var.get()

            if not summary:
                logger.error("Empty summary returned by LLM")
                raise RuntimeError("Failed to generate summary")

            summary_status = "refused" if summary.startswith("REFUSED:") else "ok"
            refusal_reason = "llm_judgment" if summary_status == "refused" else None
            if summary_status == "refused":
                logger.info(
                    f"Step 2: LLM self-refused for topic '{topic}' despite "
                    "clearing the tag-strength floor"
                )

        # Step 3: Derive the structured tags - still deterministic, no LLM.
        # Frequency-ranks the Silver tags carried in each article's metadata
        # (matched on the raw article text at Silver time) across the WIDE pool,
        # so the ranking reflects how broadly the market reported each tag.
        logger.info("Step 3: Deriving grounded tags from retrieved documents...")
        ranked_themes, ranked_risks, ranked_signals = _ranked_tags_from_documents(
            documents
        )
        key_themes = ranked_themes[: self._max_tags]
        risks = ranked_risks[: self._max_tags]
        investment_signals = ranked_signals[: self._max_tags]

        # Step 4: Extract sources from the wide pool (one URL per distinct
        # article - the analytics set already deduped by URL upstream).
        logger.info("Step 4: Extracting sources from documents...")
        sources = [doc.metadata["url"] for doc in documents if doc.metadata.get("url")]

        # Step 5: Score from the Silver tag strengths computed at Step 1; ground
        # confidence in Gold trend coverage for the thesis's categories.
        # The confidence window is the retrieval window in weeks (None = whole corpus).
        # The as-of date is the latest Gold week.
        logger.info("Step 5: Scoring (score<-Silver strengths, confidence<-Gold coverage)...")
        window_weeks = (
            None if self._retrieval_window_days <= 0
            else max(1, round(self._retrieval_window_days / 7))
        )
        # The trend repository uses the sync Supabase client, so the read runs
        # in a worker thread; awaiting it inline would block the event loop for
        # every other in-flight request. Scoped to the confidence window so the
        # Gold read stays bounded as history grows (identical result to a full
        # read).
        metrics = await asyncio.to_thread(
            self._trend_repository.fetch_recent, window_weeks
        )
        covered_weeks, window_weeks, as_of = _gold_confidence_inputs(
            documents, metrics, window_weeks
        )
        score_result = self._scoring_service.score_opportunity(
            risks=risks,
            signal_strength=signal_strength,
            theme_strength=theme_strength,
            risk_strength=risk_strength,
            covered_weeks=covered_weeks,
            window_weeks=window_weeks,
        )

        logger.info("Successfully generated structured thesis with scoring")
        return StructuredThesis(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_output=summary,
            opportunity_score=score_result["score"],
            confidence_level=score_result["confidence_level"],
            confidence_as_of=as_of,
            recommendation=score_result["recommendation"],
            key_risk_factors=score_result["key_risks"],
            summary_source=summary_source,
            summary_status=summary_status,
            refusal_reason=refusal_reason,
        )

    async def refine_thesis(
        self,
        topic: str,
        documents: List[Document],
        current_thesis: StructuredThesis,
        feedback_items: List[str],
        summary_documents: Optional[List[Document]] = None,
        prior_feedback: Optional[List[List[str]]] = None,
        theme_delta: int = 0,
        risk_delta: int = 0,
        signal_delta: int = 0,
    ) -> StructuredThesis:
        """Refine an existing thesis based on user feedback.

        Calls the LLM for a feedback-aware rewrite, then runs the same
        structuring as generate_thesis; assemble_node owns the single
        deterministic scoring step, keeping the scoring service the
        sole authority for scores.

        Args:
            topic: Original market topic for analysis.
            documents: The wide analytics pool - the distinct articles retrieval
                surfaced. Drives the displayed grounded tags (scores are frozen
                downstream, so they are not recomputed here).
            current_thesis: The thesis to refine.
            feedback_items: This round's predefined feedback strings. Drives both
                the evidence lens (which pool articles the rewrite reads) and the
                prose the LLM is asked to address.
            summary_documents: The subset the original summary read; the starting
                point the lens blends fresh evidence into. Defaults to `documents`.
            prior_feedback: Earlier rounds' feedback (oldest first), passed to the
                LLM as already-satisfied constraints to preserve while addressing
                this round.
            theme_delta: Planner-LLM-proposed cap adjustment for themes
                (-1/0/+1), clamped by _apply_cap_deltas.
            risk_delta: Same, for risks.
            signal_delta: Same, for investment signals.

        Returns:
            New StructuredThesis with refinements applied.
        """
        logger.info(f"Refining thesis for topic: {topic} based on {len(feedback_items)} feedback items")
        summary_documents = summary_documents if summary_documents is not None else documents

        # Step 1: Get refined summary from LLM using feedback. A "tag_strength_floor"
        # refusal is evidence-invariant - it is a floor on the pool, which
        # refinement never re-retrieves, so it would fail identically again; skip
        # the call and keep the refusal as-is. An "llm_judgment" refusal is a soft
        # call the LLM made on its own; new feedback (and the lens's fresh
        # evidence, below) may change it, so retry and let the prompt decide again.
        current_thesis_text = current_thesis.raw_output or ""
        summary_status = current_thesis.summary_status
        refusal_reason = current_thesis.refusal_reason
        if summary_status == "refused" and refusal_reason == "tag_strength_floor":
            logger.info("Step 1: Original summary failed the tag-strength floor; skipping rewrite")
            refined_summary = current_thesis_text
        else:
            logger.info("Step 1: Refining thesis with LLM feedback...")
            # Bias the rewrite's evidence toward what this round's feedback asks
            # for, drawn from the wide pool (see _select_feedback_evidence). Only
            # changes which docs the LLM reads; tags/score are unaffected.
            llm_docs = _select_feedback_evidence(
                documents, summary_documents, feedback_items,
                current_thesis, len(summary_documents),
            )
            # Observability only: record which lens fired and
            # how many fresh articles it pulled into the rewrite.
            lens = _lens_for_feedback(feedback_items)
            if lens:
                original_urls = {_doc_url(d) for d in summary_documents if _doc_url(d)}
                fresh = sum(
                    1 for d in llm_docs if _doc_url(d) and _doc_url(d) not in original_urls
                )
                logger.info(
                    f"Step 1: evidence lens '{lens}' pulled {fresh} fresh "
                    f"distinct article(s) into the rewrite (of {len(llm_docs)} total)"
                )
            refined_summary = await self._llm.refine(
                llm_docs, current_thesis_text, feedback_items,
                prior_feedback=prior_feedback,
            )

            if not refined_summary:
                logger.error("Empty refined summary returned by LLM")
                raise RuntimeError("Failed to refine thesis")

            summary_status = "refused" if refined_summary.startswith("REFUSED:") else "ok"
            refusal_reason = "llm_judgment" if summary_status == "refused" else None
            if summary_status == "refused":
                logger.info(f"Step 1: LLM self-refused again on refinement for topic '{topic}'")

        # Step 2: Derive grounded tags from the retrieved docs; apply the
        # quantitative effect of the feedback via the planner-LLM-proposed
        # cap deltas (clamped, see _apply_cap_deltas).
        logger.info("Step 2: Deriving feedback-adjusted grounded tags...")
        ranked_themes, ranked_risks, ranked_signals = _ranked_tags_from_documents(
            documents
        )
        key_themes, risks, investment_signals = _apply_cap_deltas(
            ranked_themes, ranked_risks, ranked_signals,
            theme_delta, risk_delta, signal_delta, self._max_tags
        )

        # Step 3: Extract sources from document metadata (same as generate_thesis)
        logger.info("Step 3: Extracting sources from documents...")
        sources = [doc.metadata["url"] for doc in documents if doc.metadata.get("url")]

        logger.info("Successfully refined thesis content (scoring handled downstream)")
        return StructuredThesis(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_output=refined_summary,
            summary_status=summary_status,
            refusal_reason=refusal_reason,
        )
