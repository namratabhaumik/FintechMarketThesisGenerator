# Limits

The boundaries you may run into, and why they exist.

## Product limits

| Limit | Value | Why |
| --- | --- | --- |
| Refinement rounds per thesis | 3 | After three grounded revisions, the leverage is in rephrasing the query, not re-polishing the same evidence. Enforced server-side. |
| Sources per thesis | Up to 50 articles | Every distinct article that clears the relevance floor is shown, so the score, tags, and confidence reflect real coverage rather than a small sample. The narrative itself is written from a smaller MMR-selected subset (relevance with a redundancy penalty) to keep it focused and cheap. Configurable via `RETRIEVAL_MAX_ARTICLES`; sparse topics surface far fewer. |
| Compare view | Current + 2 past theses | Three columns is where the side-by-side table stops being skimmable. |
| Related-theses recall floor | 86% query similarity | Recall means "same topic researched before", not "also fintech". |
| Corpus scope | Curated fintech news feeds | Tagging, scoring, and trends all assume the fintech taxonomy; general-purpose queries refuse rather than stretch. |

## Rate limits

Applied per account:

| Operation | Default |
| --- | --- |
| Thesis generation | 10 per minute |
| Refinement | 20 per minute |

Exceeding a limit returns `429` with a `rate_limit_exceeded` error; wait for the window to reset and retry. The generation and refinement endpoints are limited because each invokes the language model; reads (library, single thesis, health) are not separately limited by default.

## Hosting characteristics

The public deployment runs on free-tier infrastructure:

- **Cold starts.** After roughly 15 minutes of inactivity the backend spins down. The next request boots the service and re-downloads the embedding model, so the first generation (or library load) after idle can take a minute or two.

## Data characteristics

- **Frozen verdicts.** Articles are classified and tagged exactly once at ingestion; the corpus view behind an old thesis does not shift underneath it.
- **Point-in-time theses.** A thesis reflects the corpus at generation time. For a current view of a topic you researched before, generate a fresh thesis - [recall](guides/library-and-recall.md) will link the old one for comparison.
