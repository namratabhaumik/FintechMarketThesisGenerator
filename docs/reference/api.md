# API reference

Base URL (production):

```
https://fintechmarketthesisgenerator-prod.onrender.com/api
```

All endpoints except `GET /api/health` require a Supabase JWT:

```
Authorization: Bearer <access_token>
```

See [Authentication](../guides/auth.md) for how to obtain a token. Requests are subject to per-account rate limits ([Limits](../limits.md)).

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/theses` | Generate a thesis for a query and persist it. Returns the full job (`201`, with a `Location` header). Refuses with `422` when the corpus cannot ground the query. |
| `GET` | `/theses` | List your theses, most recent first. Supports `limit`, `offset`, and a `status` filter; admins may pass `all=true` for every user's theses. |
| `GET` | `/theses/{job_id}` | Full state of one thesis job: thesis, versions, sources, feedback, execution log, related theses. |
| `POST` | `/theses/{job_id}/refinements` | Run one refinement round with feedback reasons. Rejected once approved or after the third round. |
| `PUT` | `/theses/{job_id}/approval` | Approve the thesis (terminal; idempotent). |
| `DELETE` | `/theses/{job_id}` | Delete any user's thesis. Admin only (`204`). |
| `GET` | `/feedback-options` | The fixed set of refinement feedback reasons the app offers. |
| `GET` | `/health` | Health check. No auth. |

For full request and response schemas, use the interactive Swagger UI served by the dev environment: [https://fintechmarketthesisgenerator.onrender.com/docs](https://fintechmarketthesisgenerator.onrender.com/docs). It reflects the current API by construction (the production API is the same code with schema browsing disabled). Note the dev service is free-tier hosted, so the first load after idle can take a minute, and executing calls from Swagger still requires a bearer token.

## Error format

Errors carry a machine-readable body:

```json
{
  "detail": {
    "code": "insufficient_evidence",
    "message": "The sources retrieved for this specific query don't span themes, risks, and investment signals together. Try broadening the query."
  }
}
```

| Code | Status | Meaning |
| --- | --- | --- |
| `no_relevant_documents` | 422 | Retrieval found nothing in the corpus for this query. |
| `insufficient_evidence` | 422 | Retrieved evidence does not span all three tag dimensions; thesis refused. |
| `retrieval_failed` | 500 | Corpus retrieval errored. |
| `generation_failed` | 502 | The language model failed to produce a thesis. |
| `persistence_failed` | 500 | The operation ran but could not be saved. |
| `job_not_found` | 404 | No thesis job with that id (or not visible to this account). |
| `thesis_not_generated` | 409 | The job has no thesis to refine or approve. |
| `already_approved` | 409 | Approved theses cannot be refined. |
| `max_refinements_reached` | 409 | The three-round refinement cap is exhausted. |
| `conflict` | 409 | The job was approved or refined elsewhere while this request ran; reload. |
| `refinement_not_supported` | 501 | The configured backend cannot run the refinement agent. |
| `forbidden` | 403 | Admin role required. |
| `deletion_failed` | 500 | Admin delete errored. |
| `rate_limit_exceeded` | 429 | Too many requests; retry after the window resets. |
