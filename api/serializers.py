"""Serialization and deserialization helpers for job data.

Handles converting between Supabase JSON rows and Python domain objects
(StructuredThesis, Document) so the job manager and _RowProxy
stay focused on storage and attribute mapping.

The functions come in matched pairs that move data in opposite directions:
    serialise_* : domain object --> JSON-safe dict (writing to Supabase)
    rehydrate_* : stored JSON dict --> domain object (reading back from Supabase)
The only fields that need special handling are ones JSON cannot represent
directly (datetimes, the RefinementStatus enum); the rest pass straight through.
"""

import json
import logging
from dataclasses import asdict, fields
from datetime import date
from typing import Any, Dict, Optional

from langchain_core.documents import Document

from api.schemas import RefinementStatus
from core.models.thesis import StructuredThesis

logger = logging.getLogger(__name__)


def rehydrate_thesis(raw: Any) -> Optional[StructuredThesis]:
    """Convert a stored JSON dict to a StructuredThesis, or None.

    The stored shape is a flat dict whose keys match the dataclass fields, so it
    expands directly into the constructor. A missing/non-dict value (e.g. a job
    with no thesis yet) becomes None rather than an error. `confidence_as_of` is
    stored as an ISO 8601 date string and parsed back into a `date`, mirroring
    serialise_thesis.

    Unknown keys are dropped rather than passed through: if a dataclass field
    is ever renamed or removed, rows stored under the old schema must degrade
    to the field default instead of 500ing every read with a TypeError.
    """
    if raw and isinstance(raw, dict):
        known = {f.name for f in fields(StructuredThesis)}
        data = {k: v for k, v in raw.items() if k in known}
        as_of = data.get("confidence_as_of")
        if isinstance(as_of, str):
            data["confidence_as_of"] = date.fromisoformat(as_of)
        return StructuredThesis(**data)
    return None


def rehydrate_docs(raw: Any) -> list:
    """Convert stored JSON dicts to LangChain Document objects.

    A Document is reconstructed only from a dict that actually carries
    page_content (the shape serialise_docs emits); anything else is left as-is so
    already-hydrated or unexpected values pass through untouched.
    """
    if not raw:
        return []
    result = []
    for d in raw:
        if isinstance(d, dict) and "page_content" in d:
            result.append(
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            )
        else:
            result.append(d)
    return result


def rehydrate_query_embedding(raw: Any) -> Optional[list]:
    """Parse a stored query embedding back into a list of floats.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return [float(x) for x in json.loads(raw)]
        except (ValueError, TypeError):
            logger.warning("Skipping malformed query_embedding")
            return None
    return None


def serialise_thesis(thesis: StructuredThesis) -> dict:
    """Convert a StructuredThesis to a JSON-safe dict.

    Every field is JSON-safe via a flat asdict except `confidence_as_of`,
    which is a `date` and not natively JSON-serialisable, so it's emitted as
    an ISO 8601 string; rehydrate_thesis is the exact inverse.
    """
    data = asdict(thesis)
    if isinstance(data.get("confidence_as_of"), date):
        data["confidence_as_of"] = data["confidence_as_of"].isoformat()
    return data


def serialise_docs(docs: list) -> list:
    """Convert LangChain Documents to JSON-safe dicts.

    Each Document is flattened to {page_content, metadata} (the shape
    rehydrate_docs expects). Items that are not Documents are passed through
    unchanged, mirroring the read side.

    The transient "embedding" metadata key (carried only from retrieval into the
    in-request MMR pass) is dropped.
    """
    result = []
    for d in docs:
        if hasattr(d, "page_content"):
            metadata = {k: v for k, v in (d.metadata or {}).items() if k != "embedding"}
            result.append({"page_content": d.page_content, "metadata": metadata})
        else:
            result.append(d)
    return result


def serialise_query_embedding(embedding: Any) -> Optional[str]:
    """Format a query embedding for the `jobs.query_embedding` vector(512) column.
    """
    if embedding is None:
        return None
    if isinstance(embedding, str):
        return embedding
    return json.dumps([float(x) for x in embedding])


def serialise_job_fields(**fields) -> Dict[str, Any]:
    """Serialise arbitrary job fields for Supabase storage.

    The single entry point the job manager calls before a write. It dispatches
    each known field to its matching serialiser (thesis / docs); any other field 
    is stored verbatim, so callers can mix domain objects and plain values freely.
    """
    payload: Dict[str, Any] = {}
    for key, value in fields.items():
        if key == "thesis" and value is not None:
            payload[key] = serialise_thesis(value)
        elif key == "thesis_history":
            payload[key] = [serialise_thesis(t) for t in value]
        elif key == "refinement_status" and isinstance(value, RefinementStatus):
            payload[key] = value.value
        elif key in ("retrieved_docs", "summary_docs"):
            payload[key] = serialise_docs(value)
        elif key == "query_embedding":
            payload[key] = serialise_query_embedding(value)
        else:
            payload[key] = value
    return payload
