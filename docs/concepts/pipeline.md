# The thesis pipeline

A FinThesis thesis is the product of two pipelines: an **offline corpus pipeline** that continuously turns fintech news into tagged, embedded, trend-tracked evidence, and a **request-time pipeline** that answers your query from that corpus. Understanding the split explains most of the product's behavior - including when it refuses.

## The corpus (offline)

The corpus is built in medallion layers:

**Bronze - raw intake.** Articles arrive from curated fintech RSS feeds and are stored as-is.

**Silver - classify, tag, embed.** Each Bronze article is decided on exactly once:

- A relevance classifier keeps fintech articles and discards the rest. The verdict is recorded and never revisited, so the corpus is stable over time - a thesis you generated last month was built from the same article set view it shows today.
- Accepted articles are scraped for full text and tagged along three dimensions - **themes**, **risks**, and **investment signals** - against a fintech taxonomy. Tagging is deterministic (keyword-based with a semantic layer for concept-driven themes), so the same article always carries the same tags.
- The article text is embedded and written to the vector store with its tags and publication date in the metadata.

**Gold - trends and base rates.** The tagged corpus is aggregated two ways in a single pass:

- **Weekly trend metrics** per category, across all three dimensions - coverage volume over time. This is where the thesis confidence figure comes from: the "trends as of" date shown under Confidence is the trend window the figure was computed against.
- **Tag base rates** - what share of the whole fintech corpus carries each category. Payments-style categories sit near half the corpus; niche ones sit in low single digits. A thesis divides by these to tell "this query's evidence is unusually full of Insurtech" apart from "Insurtech articles exist", which is what keeps its tags responsive to the query rather than descriptive of the corpus. See [Why this design](#why-this-design).

Both are recomputed from Silver on every run, so they track the corpus as it grows. A category that leaves the corpus loses its base-rate row rather than lingering as a stale denominator.

## Answering a query (request time)

When you click **Generate Thesis**:

1. **Embed once.** Your query is embedded a single time; the same embedding is used for retrieval now and for [related-theses recall](../guides/library-and-recall.md) later.
2. **Retrieve a wide evidence set.** A relevance search pulls the matching article chunks from the corpus, drops anything below the similarity floor, then keeps one best chunk per article - up to ~50 distinct articles. This full set is what the score, tags, confidence, and source list are computed over, so they reflect real coverage rather than a small sample. Date expressions in the query ("since 2025", "last quarter") influence the time window. If nothing clears the floor, generation stops here with a refusal.
3. **Check the evidence.** The retrieved articles' Silver tags are counted per dimension across the whole set. If any of themes, risks, or signals has no tags at all, the thesis is refused rather than padded - see [Refusals and fallbacks](refusals.md) for why.
4. **Rank the tags.** The tags the retrieved articles carry are ranked by **lift**: the share of this evidence set carrying a tag, divided by the share of the whole corpus carrying it (the Gold base rate). A tag ranks highly for being over-represented here, not for being common. The top few per dimension become the thesis's themes, risks, and signals, each carrying the source articles behind it. A tag must appear in a minimum share of the evidence set before its lift is trusted, because a single article carrying a rare tag produces a large ratio by accident.
5. **Score.** The opportunity score (1-5) is computed from the tag strength across dimensions, and confidence from Gold trend coverage. Confidence asks how many weeks of corpus history discussed the thesis's **displayed** tags, over the retrieval window in weeks - capped at the span Gold actually holds, since a week with no data could never be covered. The recommendation (`Pursue` / `Investigate` / `Skip`) follows from the score.
6. **Write the narrative.** MMR (maximal marginal relevance) narrows the evidence set to a small, diverse subset - relevant to the query but penalized for redundancy with each other, so near-identical articles do not crowd out a dissenting source - and an LLM writes the summary from just those, specifically for your query. It is allowed to refuse if those sources do not actually address the question. The structured sections (themes, risks, signals) come from the ranked tags, not from the LLM, so they stay grounded even when the narrative is refused.
7. **Persist atomically.** The completed job - thesis, the full source set with each article's similarity to your query, the diverse subset the narrative used, and your query embedding - is saved in a single write. A failure anywhere leaves no half-written thesis in your library; you get an error instead.

## Why this design

The corpus pipeline is where the trust comes from: tags are assigned deterministically on full article text at ingestion, so when a thesis says "regulatory risk", that label traces to specific articles, not to an LLM's improvisation at request time. The LLM's job is narrow - write a narrative over evidence that already exists - and the system prefers refusing to stretching thin evidence. The result is a thesis where every layer (sources, tags, score, narrative) can be audited against the layer below it.

### Why tags rank by lift, not by count

A wide evidence set is deliberate - the score, tags and sources should reflect real coverage rather than a handful of articles. But it has a consequence that is easy to miss: at up to 50 articles, the evidence set is a substantial slice of the whole corpus. Rank tags by how often they appear in that slice and you mostly recover the corpus's own tag ordering, so the same few categories surface no matter what was asked. The number answers "what does fintech news talk about?" rather than "what is this query about?".

Dividing by the corpus base rate removes that. A category on half the corpus has to be on well over half the evidence set to look notable; a category on 5% of the corpus only needs a modest presence. What surfaces is what this query's evidence is *unusually* full of.

The minimum-source-articles bar exists because lift is unstable on small counts. One article carrying a category that appears in 0.5% of the corpus scores an enormous ratio purely by accident, and without a floor the rarest tag in the taxonomy would top every thesis. The bar scales with the evidence set (a fixed share of the displayed sources) with an absolute floor, so a genuinely narrow query is not held to a threshold it cannot structurally reach. If the bar would empty a dimension entirely, it is set aside for that dimension rather than returning nothing.

### Why confidence counts weeks, and only for the displayed tags

Confidence is a statement about the corpus, not about whether the thesis is correct. It asks how consistently, over time, the corpus discussed this thesis's topics - so a topic covered in most weeks of available history reads as well-supported, and one that appeared in a handful of scattered weeks reads as thin.

That only works if the categories being matched are specific. Measured against every tag the evidence set carries, the question becomes so broad that some tag lands in essentially every week, and the answer degenerates into "weeks that hold any data at all" - identical for every query, and stuck at its ceiling once history fills in. Scoped to the tags the thesis actually displays, a week counts only if the corpus discussed *those* categories.

The denominator is the retrieval window in weeks, capped at the span Gold holds. Uncapped, a year-long window over a younger corpus mostly measures how much history is missing. Capped, it answers the question the figure is displayed as, and it rises as ingestion fills weeks in - the same thesis re-run later can legitimately report higher confidence because the corpus behind it got denser.
