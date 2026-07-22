# Quickstart

From zero to a scored investment thesis in about two minutes.

## 1. Sign in

Open [https://finthesis-0f2c.onrender.com](https://finthesis-0f2c.onrender.com) and click **Continue with Google**. That is the whole setup - there is no separate registration. Everything you generate is private to your account.

## 2. Ask a market question

Type a fintech market topic or question into the query bar and press **Generate Thesis** (or Enter). Good first queries are market-shaped rather than encyclopedic:

- `Future of digital lending in Southeast Asia`
- `BNPL regulation and credit risk`
- `Embedded finance opportunities for SMBs`
- `AI fraud detection in payments`

Generation typically takes a few seconds once the service is warm (the very first request after idle can take longer - see [Limits](limits.md)).

## 3. Read the thesis

<iframe width="100%" height="400" src="https://www.youtube-nocookie.com/embed/Yu1gXi3OxcU"
  title="FinThesis: quickstart" frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen></iframe>

The result page, top to bottom:

| Section | What it tells you |
| --- | --- |
| **Investment Score** | Opportunity score from 1 to 5, computed from the strength of the evidence across the thesis dimensions. |
| **Confidence** | How well recent corpus trends cover this area, with the date of the trend window it was computed from ("trends as of ..."). |
| **Recommendation** | `Pursue`, `Investigate`, or `Skip`. |
| **Source Articles** | Every article the thesis is grounded in, with publication date range and a per-article "% relevant to your query" retrieval similarity. |
| **Raw Summary** | The narrative summary, written specifically to answer your query from those sources. |
| **Key Themes / Risks / Investment Signals** | The structured dimensions, each derived from tags assigned to the source articles at ingestion time. |

The URL now carries a `?job_id=...` parameter: bookmark or share it and the full thesis state restores on load.

## 4. What next

- Not satisfied with the thesis? [Refine it](guides/refining-a-thesis.md) with structured feedback (up to three rounds), then approve it.
- Got a refusal message instead of a thesis? That is deliberate - see [Refusals and fallbacks](concepts/refusals.md).
- Want the thesis outside the app? [Export it](guides/exporting.md) as text, Markdown, or PDF.
