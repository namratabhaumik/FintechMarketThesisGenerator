"""Episodic recall: rank past thesis runs by query similarity.

Each job stores the embedding of its query (`query_embedding`, set at generate
time by the embedding model). A new run embeds its query with the same model and
ranks past episodes by cosine similarity, so the user sees related prior work
instead of starting from zero. This is a query-to-query comparison, so it only
needs the query embeddings to be self-consistent (same model) - it is unrelated
to the document/chunk vector space.

Recall is done in Python (sub-millisecond at this scale).
"""

from typing import List, Tuple

import numpy as np


def recall_similar(
    query_embedding,
    jobs,
    top_n: int = 3,
    min_score: float = 0.3,
) -> List[Tuple[object, float]]:
    """Rank `jobs` by cosine similarity of their stored query embedding.

    Args:
        query_embedding: The new query's embedding (a sequence of floats).
        jobs: Past episodes, each exposing a `query_embedding` attribute. Jobs
            with no embedding (older rows), a mismatched dimension, or malformed
            data are skipped.
        top_n: Max number of matches to return.
        min_score: Cosine-similarity floor; matches below it are dropped so only
            genuinely related episodes surface.

    Returns:
        Up to `top_n` (job, score) pairs, most similar first.
    """
    if query_embedding is None or len(query_embedding) == 0:
        return []
    q = np.asarray(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []

    scored: List[Tuple[object, float]] = []
    for job in jobs:
        emb = getattr(job, "query_embedding", None)
        if not emb:
            continue
        try:
            v = np.asarray(emb, dtype=np.float32)
            if v.shape != q.shape:
                continue
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            score = float(np.dot(q, v) / (q_norm * v_norm))
        except (TypeError, ValueError):
            # Malformed embedding on this row - skip it, not the whole recall.
            continue
        if score >= min_score:
            scored.append((job, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]