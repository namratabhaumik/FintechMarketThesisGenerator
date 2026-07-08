# FinThesis

> An agentic pipeline that reads fintech news so you don't have to.

[![FinThesis Demo](https://img.youtube.com/vi/73SnVdzeVrg/hqdefault.jpg)](https://youtu.be/73SnVdzeVrg)

---

## What it does

Give FinThesis a fintech market query - "crypto in Asia", "BNPL in Europe", "embedded insurance" - and it produces a structured investment thesis built from real-time fintech news.

Each thesis includes:

- A pattern-structured breakdown: key themes, risks, and investment signals mapped to a fintech taxonomy
- A rule-based opportunity score (0–5), confidence level, and Pursue / Investigate / Skip recommendation
- The source articles used to reach the conclusion
- An optional refinement loop - if the output isn't right, pick from a fixed set of feedback reasons and a LangGraph agent rewrites the thesis via tool calls

## How it works

**Medallion pipeline (Bronze → Silver → Gold).** Ingestion is a three-stage medallion, run daily so the corpus accumulates real history:

- **Bronze** lands raw RSS entries verbatim into `articles_raw`, deduped by URL - no classifying, scraping, or embedding. It just accumulates articles on the published-at axis (RSS only carries recent items, so depth comes from polling forward over time).
- **Silver** does all the enrichment: it classifies each Bronze article's fintech relevance (LLM), and for the accepted ones scrapes the full text, tags it (themes / risks / signals, matched on word boundaries against a fintech taxonomy), and embeds it (the article **title + full scraped body**, split into chunks; the thin RSS summary is only used for classification) into Supabase pgvector via FastEmbed (ONNX). Only the fintech-accepted subset is scraped and embedded; each verdict is frozen point-in-time. Every article ends in one bucket:

  ```
  classify ─ error ──────────────▶ errored      (retry next run - classifier failure)
           ├ not fintech ────────▶ verdict NO   (frozen)
           └ fintech ─ scrape ─── fail ──▶ errored      (retry next run - scraper failure)
                                ├ fail ──▶ quarantined  (dead-letter, no retry)
                                └ success ─▶ verdict YES + document (frozen)
  ```

- **Gold** aggregates the accepted corpus into per-(week, category) trend metrics across all three tag dimensions - coverage volume over time, the trend signal.

**Retrieval → generation.** A query runs MMR retrieval over the Silver embeddings and passes the selected chunks to Gemini for summarization and structuring. The opportunity **score** is derived from the Silver tag strengths; the **confidence** is grounded in Gold trend coverage for the thesis's categories; and the **recommendation** (Pursue / Investigate / Skip) follows from the score via fixed thresholds.

**Recency window.** Retrieval only considers articles published within a trailing window (default: the last year), so a thesis reflects a trend over that window rather than spot news. The window slides with the query date and is editable via `RETRIEVAL_WINDOW_DAYS` (`0` searches the whole corpus). 

**Refinement agent (LangGraph):** After reading a thesis, the user picks from a fixed set of feedback reasons. A LangGraph agent reasons about which tool to call and rewrites only the part of the thesis that needs changing.

- **`InjectedState` for tool context.** Tools need the current thesis, documents, and feedback history but the LLM only specifies a lightweight intent argument. `InjectedState` injects the full graph state into each tool at call time without exposing it to the LLM.
- **Hallucination detection on message history.** Validates the structured `tool_calls` field on `AIMessage` objects against the registered tool registry. Invented tool names are caught in the trace before reaching the user.
- **Execution trace in the UI.** Each tool invocation is logged to `execution_log` in graph state and rendered in the UI (tool name, status, refinement round). The agent's behavior is inspectable, not a black box.
- **Fixed feedback reasons.** Each reason maps directly to a tool, keeping the agent's decision space narrow and its routing predictable.

**Approval.** Once a thesis is right, the user clicks Approve. Approval is terminal: it stamps the time, freezes the run, and drops it out of the resume picker (nothing left to refine).

**Session persistence and resume.** Every run is checkpointed to a Supabase job row - the thesis, retrieved docs, refinement count and status, feedback history, execution log, and query embedding - after generation, after each refinement round, and on approval. This makes a run recoverable across a refresh, a new tab, or a server restart. There are three ways back into a past run:

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

