# FinThesis

FinThesis is an AI research assistant for fintech markets. You give it a market topic or question; it retrieves recent articles from a continuously ingested fintech news corpus and returns a **scored investment thesis**: key themes, risks, investment signals, an opportunity score, a confidence level, and a recommendation - every part grounded in the sources it shows you.

*FinThesis is for informational and research purposes only and does not constitute financial or investment advice.*

**Live app:** [https://finthesis-0f2c.onrender.com](https://finthesis-0f2c.onrender.com)

<!-- Clip: assets/clips/overview.mp4
<video controls muted playsinline width="100%">
  <source src="assets/clips/overview.mp4" type="video/mp4">
</video>
-->

## What it does

- **Generates theses from evidence.** Every thesis is built from articles retrieved out of a curated fintech corpus. Each source is listed with its publication date and how relevant it is to your query.
- **Refuses rather than guesses.** If the corpus has nothing relevant, or the retrieved evidence cannot ground all three thesis dimensions (themes, risks, signals), FinThesis declines to produce a partial thesis and tells you why. See [Refusals and fallbacks](concepts/refusals.md).
- **Iterates with you.** A refinement agent revises a thesis based on structured feedback, up to three rounds, with a visible execution trace and version history. See [Refining a thesis](guides/refining-a-thesis.md).
- **Remembers your research.** Past theses live in a browsable library, and new theses automatically surface closely related past ones for side-by-side comparison. See [Library, resume and recall](guides/library-and-recall.md).

## Where to start

- New here: the [Quickstart](quickstart.md) takes you from sign-in to your first thesis.
- Curious how a thesis is built: [The thesis pipeline](concepts/pipeline.md).
- Calling the API directly: [Authentication](guides/auth.md) and the [API reference](reference/api.md).

!!! note "Free-tier hosting"
    The demo runs on free-tier infrastructure. After a period of inactivity the first request can take a minute or two while the backend cold-starts; subsequent requests are fast. See [Limits](limits.md).
