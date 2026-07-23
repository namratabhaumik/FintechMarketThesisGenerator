# Library, resume and recall

Everything you generate is kept, browsable, and cross-referenced. Three features build on that: the past-theses library, the resume picker, and related-theses recall.

## Past theses

The **Past theses** section below the results lists your previous runs, most recent first, ten per page: query, score, date, status (`approved`, `refining`, `escalated`), and recommendation. Click any row to load that thesis in full - sources, versions, execution trace and all.

The thesis you are currently viewing is excluded from its own list, so the library never points you at the page you are already on.

## Shareable URLs

Every thesis has a stable URL of the form `?job_id=...`, set automatically when a thesis is generated or opened. Refreshing restores the full state; sharing the link with yourself on another device works the same way (the recipient must be signed in as you - theses are private per account).

## Resuming a refinement

If you left a thesis mid-refinement (refined at least once, not yet approved), the query bar area offers **Resume a previous refinement** with a dropdown of resumable runs, each labeled with its query, round count, and date. Resuming restores the thesis exactly where you left it, remaining rounds included. Approved theses and untouched ones are not listed.

## Related past theses (recall)

When a new thesis is generated, FinThesis compares your query against the queries of your past theses (by embedding similarity, computed in the database) and surfaces close matches in a **Related past theses** panel: each with its score, date, recommendation, approval status, and a "% match" figure.

The similarity floor is high on purpose (86%): a related entry means "you have researched this same topic or a directly adjacent sub-topic before", not merely "also fintech". For instance, a query about payments fraud will not surface your digital-lending theses.

## Comparing theses

<iframe width="100%" height="400" src="https://www.youtube-nocookie.com/embed/umBMsORM0Nw"
  title="FinThesis: comparing theses" frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen></iframe>

Tick up to two related theses and click **Compare with current** for a side-by-side table: date, score, confidence, recommendation, themes, risks, and signals as rows, one column per thesis. The current thesis is always the first column. The cap of three columns total keeps the table readable; with two past theses selected, the remaining checkboxes disable themselves.
