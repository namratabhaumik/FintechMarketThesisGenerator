# FinThesis

> An agentic pipeline that reads fintech news so you don't have to.

**Live:** [finthesis-0f2c.onrender.com](https://finthesis-0f2c.onrender.com)

---

## What it does

Give FinThesis a fintech market query - "crypto in Asia", "BNPL in Europe", "embedded insurance" - and it produces a structured investment thesis built from real-time fintech news.

Each thesis includes:

- A pattern-structured breakdown: key themes, risks, and investment signals mapped to a fintech taxonomy
- A rule-based opportunity score (0–5), confidence level, and Pursue / Investigate / Skip recommendation
- The source articles used to reach the conclusion
- An optional refinement loop - if the output isn't right, pick from a fixed set of feedback reasons and a LangGraph agent rewrites the thesis via tool calls

## What's in a thesis?

Every output you see maps to a specific path. The tag on each says *how* it is computed: **[LLM]** a model call, **[Agent]** the LangGraph refinement agent, **[Python]** deterministic code with no model involved.

Almost everything starts from the same step: a query runs **[Python]** relevance retrieval over the Silver pgvector embeddings, applies the similarity floor, then dedupes to **distinct articles** (one best chunk per URL, up to ~50) - the **wide evidence set**, each article carrying its Silver tags + metadata. Tags, score, confidence, and the source list are all computed over this full set, so they reflect real coverage rather than a small sample. A separate **[Python]** MMR step then narrows the set to a small, diverse subset that the summary LLM actually reads. `create_thesis` in [`api/routes.py`](api/routes.py) → [`core/services/retrieval_service.py`](core/services/retrieval_service.py) (`retrieve` + `select_diverse`).

**Two independent refusal gates**, both deterministic and evidence-driven rather than left to model judgment:

1. **Whole-thesis gate (post-retrieval check).** If the retrieved articles don't carry Silver tags in **all three** dimensions (themes, risks, signals) across the wide evidence set - i.e. any dimension's tag count is `0` - the request is refused with a `422` (`insufficient_evidence`) *before* the model is ever called. The missing dimensions are logged. `create_thesis` in [`api/routes.py`](api/routes.py).
2. **Summary-only gate (pre-LLM-call), a thin safety net.** Passing gate 1 only guarantees each dimension has *some* tag coverage across the wide set - as little as one incidental hit. Before calling the LLM, `generate_thesis` checks the Silver tag strengths **of the diverse subset the LLM will actually read** against a stricter floor (`theme_strength ≥ 3`, `risk_strength ≥ 2`, `signal_strength ≥ 2`). Measuring the floor on that small subset (not the wide set) keeps the thresholds calibrated to what the model sees. Below the floor, the LLM call is skipped entirely and the summary is marked **refused** - the rest of the thesis (tags, score, confidence, all computed over the wide set) still renders normally. `MIN_THEME_STRENGTH_FOR_SUMMARY` etc. in [`thesis_generator_service.py`](core/services/thesis_generator_service.py) (will be re-calibrated against corpus distribution on 30-day daily `corpus_probe`).

Even past gate 2, the LLM prompt itself carries a `"respond with REFUSED:"` instruction as a secondary check (**gate 3**) - it can decline for query-nuance reasons the tag count can't see (e.g. tags are broadly "fintech" but don't cover the query's specific angle). This one *is* left to model judgment, since it's catching what the deterministic gates structurally can't. The UI's **summary_status** ("ok"/"refused") reflects the outcome of whichever gate fired, if any; a **refusal_reason** field (`"tag_strength_floor"` or `"llm_judgment"`) records *which* one, distinguishing an evidence-invariant refusal from a model judgment call (the Raw Summary panel displays the message).

*Why gate 2 exists beyond being a safety net:* before it, the summary and the structured half of the thesis ran on disjoint evidence - the narrative was purely by the model's own reading of raw article text, while themes/risks/signals/score/confidence were derived entirely from Silver tag metadata. Those were two independent judgments that could (and did) disagree (eg: a thesis could carry a confidently-refused summary alongside a fully-populated, well-scored structured half). Gate 2 makes the *decision to even attempt a narrative* depend on the Silver tag strength of the very docs the model would read, so thin evidence in front of the model consistently means "no narrative attempt". Gate 3 still sits on top as an independent text-level judgment, since tag counts can't verify the narrative addresses the query's specific angle, only that there's roughly enough underlying material to plausibly try.

- **Summary** — the MMR-selected diverse subset → **[LLM]** `llm.summarize` → narrative prose (or **[Python]** a deterministic refusal - see the two gates above). The *only* model call in generation: Gemini normally, falling back to a local extractive summarizer if Gemini is unavailable (the local path has its own deterministic topic-term refusal check, independent of the Silver tag-strength gate). A **summary_source** badge in the UI shows which one produced the text. [`thesis_generator_service.py`](core/services/thesis_generator_service.py)
- **Themes / Risks / Signals** — the wide set's articles' Silver-tag metadata → **[Python]** frequency-rank (`Counter.most_common`) → top 3 per dimension at generation (refinement can nudge this cap to 4; see below). One vote per article (dedup upstream), so the ranking reflects how broadly the market reported each tag. Tags are assigned at Silver ingest, not invented by the model. `_ranked_tags_from_documents`
- **Score** (0–5) — total Silver signal/theme/risk tag counts across the wide set → **[Python]** self-normalizing formula (signals + ½·themes lift, risks pull down) → clamp to 0–5. `_compute_score` in [`opportunity_scoring_service.py`](finthesis_internal/opportunity_scoring_service.py)
- **Confidence** (0–1) — the wide set's tag categories → matched against **Gold** weekly trend metrics → **[Python]** `covered_weeks / window_weeks`. Grounded in corpus trend coverage, not the model. `_gold_confidence_inputs` → `_compute_confidence`
- **Recommendation** — score → **[Python]** fixed thresholds (≥3.75 Pursue, ≥2.5 Investigate, else Skip). `_get_recommendation`
- **Sources** — the wide distinct-article set → **[Python]** `doc.metadata["url"]` passthrough (already one per article), in relevance order, up to ~50. `_sources_from_docs` in [`api/routes.py`](api/routes.py)
- **Related past theses** — this run's query embedding → **[Python]** pgvector `match_jobs` RPC (cosine ≥ 0.86) → past jobs ranked by *semantic similarity*. Shown inside the thesis card. `_related_for` in [`api/routes.py`](api/routes.py)
- **Past theses** — **[Python]** `GET /api/theses` → `list_jobs` ordered by `created_at DESC` (pure recency, no similarity) → the browsable library on the main page. `list_jobs` in [`api/supabase_job_manager.py`](api/supabase_job_manager.py)
- **Execution trace** (refinement only) — **[Agent]** planner LLM picks a tool → ToolNode executes → `assemble_node` diffs the old vs new thesis and logs each event → `execution_log`. [`refinement_graph.py`](core/agents/refinement_graph.py)

On **refinement**, the numbers are recomputed the same deterministic way; the planner LLM merely nudges *how many* tags surface (±1 per dimension, clamped). `refine_thesis` → `_apply_cap_deltas`. The narrative is re-written by **[LLM]** `llm.refine` - except when the current thesis's `refusal_reason` is `"tag_strength_floor"`, the rewrite is skipped. A `refusal_reason` of `"llm_judgment"` (or a healthy thesis) *does* call `llm.refine` - new feedback may change a judgment call. `llm.refine`'s prompt carries the same `"REFUSED:"` instruction as `llm.summarize`, so a retried rewrite can honestly decline again instead of fabricating prose.

## How it works

**Medallion pipeline (Bronze → Silver → Gold).** Ingestion is a three-stage medallion, run daily so the corpus accumulates real history:

- **Bronze** lands raw RSS entries verbatim into `articles_raw`, deduped by URL - no classifying, scraping, or embedding. It just accumulates articles on the published-at axis (RSS only carries recent items, so depth comes from polling forward over time).
- **Silver** does all the enrichment: it classifies each Bronze article's fintech relevance (LLM), and for the accepted ones scrapes the full text, tags it (themes / risks / signals, matched on word boundaries against a fintech taxonomy), and embeds it (the article **title + full scraped body**, split into chunks; the thin RSS summary is only used for classification) into Supabase pgvector via FastEmbed (ONNX). Themes additionally get an embedding-based semantic pass (reusing the same FastEmbed model): it adds a few concept-driven categories keywords structurally miss (e.g. *Capital Markets & Trading Tech* on a prediction-market or quant story) and vetoes lone-keyword false positives, while risks / signals stay pure keyword. Any embedding failure degrades to the keyword result. Only the fintech-accepted subset is scraped and embedded; each verdict is frozen point-in-time. Every article ends in one bucket:

  ```
  classify ─ error ──────────────▶ errored      (retry next run - classifier failure)
           ├ not fintech ────────▶ verdict NO   (frozen)
           └ fintech ─ scrape ─── fail ──▶ errored      (retry next run - scraper failure)
                                ├ fail ──▶ quarantined  (dead-letter, no retry)
                                └ success ─▶ verdict YES + document (frozen)
  ```

- **Gold** aggregates the accepted corpus into per-(week, category) trend metrics across all three tag dimensions - coverage volume over time, the trend signal.

**Retrieval → generation.** A query runs relevance retrieval over the Silver embeddings and dedupes to a wide set of distinct articles; an MMR step then passes a small, diverse subset of them to Gemini for summarization. The opportunity **score** is derived from the Silver tag strengths over the wide set; the **confidence** is grounded in Gold trend coverage for the thesis's categories; and the **recommendation** (Pursue / Investigate / Skip) follows from the score via fixed thresholds.

**Recency window.** Retrieval only considers articles published within a trailing window (default: the last year), so a thesis reflects a trend over that window rather than spot news. The window slides with the query date and is editable via `RETRIEVAL_WINDOW_DAYS` (`0` searches the whole corpus). 

**Refinement agent (LangGraph):** After reading a thesis, the user picks from a fixed set of feedback reasons. A LangGraph agent reasons about which tool to call and rewrites only the part of the thesis that needs changing.

- **`InjectedState` for tool context.** Tools need the current thesis, documents, and feedback history but the LLM only specifies a lightweight intent argument. `InjectedState` injects the full graph state into each tool at call time without exposing it to the LLM.
- **Hallucination detection on message history.** Validates the structured `tool_calls` field on `AIMessage` objects against the registered tool registry. Invented tool names are caught in the trace before reaching the user.
- **Execution trace in the UI.** Each tool invocation is logged to `execution_log` in graph state and rendered in the UI (tool name, status, refinement round). The agent's behavior is inspectable, not a black box.
- **Fixed feedback reasons.** Each reason maps directly to a tool, keeping the agent's decision space narrow and its routing predictable.
- **Feedback-aware evidence (deterministic).** Refinement never re-retrieves, but the evidence-seeking reasons ("stronger evidence for themes", "signals too vague", "missing recent trends", "too broad") re-focus *which* articles from the stored wide pool the rewrite reads - via a deterministic lens over Silver tags / `published_at` / query similarity (no embeddings/thresholds). It only changes the narrative's evidence; tags and the frozen score are untouched. Earlier rounds' feedback is passed to the rewrite as constraints to preserve, so a new round doesn't regress prior ones. `_select_feedback_evidence` + `FEEDBACK_LENS` in [`thesis_generator_service.py`](core/services/thesis_generator_service.py) / [`config/settings.py`](config/settings.py).

**Approval.** Once a thesis is right, the user clicks Approve. Approval is terminal: it stamps the time, freezes the run, and drops it out of the resume picker (nothing left to refine).

**Session persistence and resume.** Every run is checkpointed to a Supabase job row - the thesis, the wide retrieved-docs set, the MMR subset the narrative used (`summary_docs`), refinement count and status, feedback history, execution log, and query embedding - after generation, after each refinement round, and on approval. This makes a run recoverable across a refresh, a new tab, or a server restart. There are three ways back into a past run:

- **By URL.** Each run carries a `?job_id=` query param; opening that URL rehydrates the full session from its checkpoint.
- **Resume picker.** When you don't have the exact URL, a dropdown lists runs still mid-refinement so you can pick one and continue.
- **From a related thesis.** Clicking a recalled past thesis (see below) opens it via its `job_id`.

Supabase is required: the jobs table is the single state carrier, so generation, refinement, approval, and all three resume paths run through it.

**Episodic recall.** On each run the query embedding is compared against past runs, and the most similar prior theses are surfaced in a "Related past theses" panel (with score, recommendation, and similarity). It links back to those runs so prior analysis on a similar topic is one click away.

**Rule-based opportunity scoring.** Score (0–5), confidence, and recommendation come from a deterministic formula weighted by detected themes, risks, and signals.

**Dual-mode summarization.** Gemini or a local keyword-scored extractive summarizer, a no-API fallback for when rate limits or cost constraints apply.

**Langfuse observability.** Every graph run, tool call, and LLM call is traced end-to-end via a callback handler wired at the graph level.

## Constraints

Deliberate scope boundaries of the current platform:

- **Single news source.** Articles come only from TechCrunch's RSS feeds. Multi-source ingestion, and detecting or adapting to a source changing its feed format, are out of scope - the pipeline assumes the current feed structure.
- **Point-in-time history, not restated.** Raw articles are retained indefinitely and each article's fintech classification is recorded once and frozen. Changing the classifier model applies to new articles only - past records are not retroactively re-classified, so historical trends reflect what was judged at the time.

## Deployment

Frontend and backend are decoupled and deployed as separate Render services, across two environments (development and production).

**Branch flow.** `feature → develop → master`. `develop` drives the development environment ([finthesis.onrender.com](https://finthesis.onrender.com)); `master` drives production ([finthesis-0f2c.onrender.com](https://finthesis-0f2c.onrender.com)).

**CI/CD (GitHub Actions).** On every push, a read-only job runs the tests and builds the vanilla-TS frontend bundle with that environment's API base URL compiled in. A separate job (with write access) then publishes just the built static assets to a per-environment deploy branch (`deploy-dev` / `deploy-prod`) that the corresponding Render Static Site serves. The FastAPI backend deploys per source branch. A failing test skips the deploy entirely.

**API documentation.** Interactive Swagger docs at `/docs` and ReDoc at `/redoc` are gated behind the `ENABLE_DOCS` flag (enabled locally and on dev - [fintechmarketthesisgenerator.onrender.com/docs](https://fintechmarketthesisgenerator.onrender.com/docs)).

