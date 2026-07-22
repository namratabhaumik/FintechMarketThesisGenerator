# Refusals and fallbacks

FinThesis refuses to answer more readily than most AI tools, and that is its most deliberate design choice: **a partial or ungrounded thesis is worse than no thesis**, because it looks exactly like a trustworthy one. There are three distinct refusal behaviors plus one fallback. Knowing which one you hit tells you what to do next.

<iframe width="100%" height="400" src="https://www.youtube-nocookie.com/embed/tAQhZe6VqRE"
  title="FinThesis: refusals and fallbacks" frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen></iframe>

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

Sometimes a thesis *is* generated, but the Raw Summary section contains a refusal instead of a narrative. The retrieved evidence passed the deterministic dimension check, but the summarizer judged it could not honestly answer your *specific question* in prose. The structured sections below come from ingestion-time tags rather than from the LLM, so they remain trustworthy and are shown anyway. There are two flavors, and they call for different next steps.

**3a. The LLM declined on its own judgment:**

> "The sources touch on related fintech topics but don't specifically address this query - but the N themes, N risks and N signals below are grounded in the same sources and worth reviewing directly."

This is a soft, per-attempt call - the same evidence can read differently to the model on a second pass, especially with feedback attached. **What to do:** read the structured sections directly, sharpen/modify the query for a fresh thesis, *or* [refine](../guides/refining-a-thesis.md) the existing one - the refinement agent retries the narrative with your feedback in the prompt, and for evidence-seeking feedback also draws additional relevant articles from the retrieved pool, so it can generate a summary where the first pass declined.

**3b. The evidence itself is thin:**

> "The sources didn't give us enough to write a reliable narrative for this query - but the ... below are grounded in the same sources and worth reviewing directly."

This is a deterministic floor checked before the narrative attempt, evaluated on the subset the model would read. A refinement round on this refusal skips the rewrite entirely - the floor reflects the retrieved pool, which is never re-fetched - so it cannot change the outcome; the round makes no changes and says so explicitly. **What to do:** read the structured sections directly, or broaden/reframe the query to retrieve richer evidence.

## Fallback: local summarizer

If the LLM is unavailable, an amber banner appears above the summary:

> "Generated without an LLM (local extractive summarizer) - narrative quality may be reduced."

This is a degraded mode, not a refusal: the narrative is stitched from extracted source sentences, and scores and structured sections are unaffected. Regenerating later, when the LLM is reachable, produces a normal narrative.

## Summary table

| Behavior | Thesis created? | Root cause | Your move |
| --- | --- | --- | --- |
| No relevant documents | No | Corpus has nothing on this | Different or broader topic |
| Insufficient tagged evidence | No | Evidence does not span themes + risks + signals | Broaden or reframe |
| Narrative refusal (LLM judgment) | Yes | Model declined this specific question on its own | Use the structured sections; sharpen the query, or refine - retrying can succeed |
| Narrative refusal (thin evidence) | Yes | Deterministic floor on the retrieved evidence | Use the structured sections; refining will not help - broaden or reframe |
| Local summarizer | Yes | LLM unavailable | Regenerate later if narrative matters |
