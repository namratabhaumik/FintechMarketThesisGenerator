# Refusals and fallbacks

FinThesis refuses to answer more readily than most AI tools, and that is its most deliberate design choice: **a partial or ungrounded thesis is worse than no thesis**, because it looks exactly like a trustworthy one. There are three distinct refusal behaviors plus one fallback. Knowing which one you hit tells you what to do next.

<!-- Clip: assets/clips/refusals.mp4
<video controls muted playsinline width="100%">
  <source src="../assets/clips/refusals.mp4" type="video/mp4">
</video>
-->

## 1. No relevant documents

> "No relevant documents found for this query. Try a broader or different fintech topic."

Retrieval found nothing in the corpus for your query. Typical for non-fintech queries or topics the news corpus has not covered. No thesis is created.

**What to do:** rephrase toward a fintech market question, or broaden the topic.

## 2. Insufficient tagged evidence

> "Not enough tagged evidence to build a complete thesis for this query."

Retrieval *did* find articles, but their tags do not span all three thesis dimensions - somewhere among themes, risks, and investment signals there is no evidence at all. Rather than render a blank section and fall back to a made-up neutral score, FinThesis refuses before calling the LLM. No thesis is created.

This typically happens on narrow or emerging sub-topics where the corpus has coverage but the taxonomy does not (yet) have matching categories across all dimensions.

**What to do:** broaden the query, or reframe it toward the established market it belongs to.

## 3. Narrative refusal (thesis still delivered)

Sometimes a thesis *is* generated, but the Raw Summary section contains a refusal instead of a narrative, in one of two flavors:

> "The sources touch on related fintech topics but don't specifically address this query - but the N themes, N risks and N signals below are grounded in the same sources and worth reviewing directly."

> "The sources didn't give us enough to write a reliable narrative for this query - but the ... below are grounded in the same sources and worth reviewing directly."

The retrieved evidence passed the dimension check, but the summarizer judged it could not honestly answer your *specific question* in prose. The structured sections come from ingestion-time tags rather than from the LLM, so they remain trustworthy and are shown anyway.

**What to do:** read the themes, risks, and signals directly - they are the grounded core - or sharpen the query so the narrative can commit.

## Fallback: local summarizer

If the LLM is unavailable, an amber banner appears above the summary:

> "Generated without an LLM (local extractive summarizer) - narrative quality may be reduced."

This is a degraded mode, not a refusal: the narrative is stitched from extracted source sentences, and scores and structured sections are unaffected. Regenerating later, when the LLM is reachable, produces a normal narrative.

## Summary table

| Behavior | Thesis created? | Root cause | Your move |
| --- | --- | --- | --- |
| No relevant documents | No | Corpus has nothing on this | Different or broader topic |
| Insufficient tagged evidence | No | Evidence does not span themes + risks + signals | Broaden or reframe |
| Narrative refusal | Yes | Sources do not address the specific question | Use the structured sections; sharpen the query |
| Local summarizer | Yes | LLM unavailable | Regenerate later if narrative matters |
