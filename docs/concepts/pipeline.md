# The thesis pipeline

A FinThesis thesis is the product of two pipelines: an **offline corpus pipeline** that continuously turns fintech news into tagged, embedded, trend-tracked evidence, and a **request-time pipeline** that answers your query from that corpus. Understanding the split explains most of the product's behavior - including when it refuses.

## The corpus (offline)

The corpus is built in medallion layers:

**Bronze - raw intake.** Articles arrive from curated fintech RSS feeds and are stored as-is.

**Silver - classify, tag, embed.** Each Bronze article is decided on exactly once:

- A relevance classifier keeps fintech articles and discards the rest. The verdict is recorded and never revisited, so the corpus is stable over time - a thesis you generated last month was built from the same article set view it shows today.
- Accepted articles are scraped for full text and tagged along three dimensions - **themes**, **risks**, and **investment signals** - against a fintech taxonomy. Tagging is deterministic (keyword-based with a semantic layer for concept-driven themes), so the same article always carries the same tags.
- The article text is embedded and written to the vector store with its tags and publication date in the metadata.

**Gold - trends.** The tagged corpus is aggregated into per-category weekly trend metrics across all three dimensions. This is where the thesis confidence figure comes from: the "trends as of" date shown under Confidence is the trend window the figure was computed against.

## Answering a query (request time)

When you click **Generate Thesis**:

1. **Embed once.** Your query is embedded a single time; the same embedding is used for retrieval now and for [related-theses recall](../guides/library-and-recall.md) later.
2. **Retrieve.** MMR (maximal marginal relevance) search selects the most relevant article chunks from the corpus - relevant to the query but penalized for redundancy with each other, so five near-identical articles do not crowd out a dissenting source. Date expressions in the query ("since 2025", "last quarter") influence the time window. If nothing relevant is found, generation stops here with a refusal.
3. **Check the evidence.** The retrieved chunks' Silver tags are counted per dimension. If any of themes, risks, or signals has no tags at all, the thesis is refused rather than padded - see [Refusals and fallbacks](refusals.md) for why.
4. **Score.** The opportunity score (1-5) is computed from the tag strength across dimensions, and confidence from the Gold trend coverage of the queried area. The recommendation (`Pursue` / `Investigate` / `Skip`) follows from the score.
5. **Write the narrative.** An LLM writes the summary specifically for your query from the retrieved sources - and is allowed to refuse if the sources do not actually address the question. The structured sections (themes, risks, signals) come from the tags, not from the LLM, so they stay grounded even when the narrative is refused.
6. **Persist atomically.** The completed job - thesis, sources with their similarity to your query, embeddings - is saved in a single write. A failure anywhere leaves no half-written thesis in your library; you get an error instead.

## Why this design

The corpus pipeline is where the trust comes from: tags are assigned deterministically on full article text at ingestion, so when a thesis says "regulatory risk", that label traces to specific articles, not to an LLM's improvisation at request time. The LLM's job is narrow - write a narrative over evidence that already exists - and the system prefers refusing to stretching thin evidence. The result is a thesis where every layer (sources, tags, score, narrative) can be audited against the layer below it.
