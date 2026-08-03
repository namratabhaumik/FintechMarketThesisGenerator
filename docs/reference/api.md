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

### Annotations

Notes on a thesis: an anchored highlight (a root) and the replies under it. See [Annotating a thesis](../guides/annotating.md) for what they are for.

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/theses/{job_id}/annotations` | Every annotation on the thesis, roots and replies together, so a thread needs no second call. Optional `version` scopes it to one version. `404` if the thesis is not visible to you. |
| `POST` | `/theses/{job_id}/annotations` | Create a root or a reply (`201`). A root requires `version`, `section`, `start_offset`, `end_offset` and `quote`; a reply requires `parent_id` and must carry none of those. |
| `PATCH` | `/annotations/{annotation_id}` | Edit a comment's text. Author only. |
| `PATCH` | `/annotations/{annotation_id}/resolution` | Tick a thread (`"resolution": "accepted"`) or reopen it (`"resolution": null`). Roots only. Open to the thesis owner and admins, not just the author, because resolving is a state change rather than an edit. |
| `DELETE` | `/annotations/{annotation_id}` | Delete a comment; replies cascade (`204`). The author or the thesis owner. |

`section` is one of `raw_summary`, `key_themes`, `risks`, `investment_signals`. Offsets are character positions in that section's plain text.

An annotation is pinned to the thesis version it was written against: a version's text is frozen once superseded, which is what keeps the offsets valid. Refining therefore produces a version with no annotations on it.

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
| `forbidden` | 403 | Admin role required, or the annotation is not yours to edit or resolve. |
| `deletion_failed` | 500 | A delete errored. |
| `not_found` | 404 | No annotation with that id, or it is a reply where a root is required. |
| `invalid_annotation` | 400 | The annotation's shape is wrong: a reply carrying an anchor, a root missing anchor fields, a nested reply, or a version that does not exist. |
| `invalid_resolution` | 400 | Unrecognised resolution value. |
| `rate_limited` | 429 | Too many requests; retry after the window resets. |
