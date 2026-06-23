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

**Ingestion → retrieval → generation:** RSS articles fetched, embedded via FastEmbed (ONNX), and persisted in Supabase pgvector. A query runs MMR retrieval over the corpus and passes the selected chunks to Gemini for summarization and structuring.

**Recency window.** Retrieval only considers articles published within a trailing window (default: the last year), so a thesis reflects a trend over that window rather than spot news. The window slides with the query date and is editable via `RETRIEVAL_WINDOW_DAYS` (`0` searches the whole corpus). 

**Refinement agent (LangGraph):** After reading a thesis, the user picks from a fixed set of feedback reasons. A LangGraph agent reasons about which tool to call and rewrites only the part of the thesis that needs changing.

- **`InjectedState` for tool context.** Tools need the current thesis, documents, and feedback history but the LLM only specifies a lightweight intent argument. `InjectedState` injects the full graph state into each tool at call time without exposing it to the LLM.
- **Hallucination detection on message history.** Validates the structured `tool_calls` field on `AIMessage` objects against the registered tool registry. Invented tool names are caught in the trace before reaching the user.
- **Execution trace in the UI.** Each tool invocation is logged to `execution_log` in graph state and rendered in the UI (tool name, status, refinement round). The agent's behavior is inspectable, not a black box.
- **Fixed feedback reasons.** Each reason maps directly to a tool, keeping the agent's decision space narrow and its routing predictable.

**Rule-based opportunity scoring.** Score (0–5), confidence, and recommendation come from a deterministic formula weighted by detected themes, risks, and signals.

**Dual-mode summarization.** Gemini or a local keyword-scored extractive summarizer, a no-API fallback for when rate limits or cost constraints apply.

**Langfuse observability.** Every graph run, tool call, and LLM call is traced end-to-end via a callback handler wired at the graph level.

## Constraints

Deliberate scope boundaries of the current platform:

- **Single news source.** Articles come only from TechCrunch's RSS feeds. Multi-source ingestion, and detecting or adapting to a source changing its feed format, are out of scope - the pipeline assumes the current feed structure.
- **Point-in-time history, not restated.** Raw articles are retained indefinitely and each article's fintech classification is recorded once and frozen. Changing the classifier model applies to new articles only - past records are not retroactively re-classified, so historical trends reflect what was judged at the time.

