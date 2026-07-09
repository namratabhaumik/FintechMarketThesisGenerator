"""Daily read-only probe of retrieval health -> corpus_probe table.

For a fixed panel of representative fintech queries plus off-topic controls, this
embeds each query and reads the query-to-chunk similarity distribution the
match_documents RPC already returns, alongside corpus size. It writes one row per
query to corpus_probe. NOTHING is modified in the corpus - it only reads and logs.

Prereqs:
    - sql/corpus_probe.sql applied (the corpus_probe table)
    - SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY + EMBEDDING_MODEL in env

Run from the repo root:
    python -m scripts.inspect_corpus
"""

import logging
import statistics as st
from typing import Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

from config.settings import AppConfig
from dependency_injection.container import ServiceContainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# on-topic + sub-topics queries
FINTECH_QUERIES = [
    "embedded finance and banking-as-a-service",
    "buy now pay later and consumer credit",
    "stablecoins and crypto payments",
    "AI underwriting in lending",
    "neobanks and digital banking",
    "payments infrastructure",
    "regtech and compliance",
    "wealthtech and retail investing",
]
CONTROL_QUERIES = [
    "best hiking trails",
    "install custom graphics driver for my PC",
]


def _count(client, table: str) -> int:
    """Row count of `table` without pulling any rows (head + exact count)."""
    resp = client.table(table).select("*", count="exact", head=True).execute()
    return resp.count or 0


def _distribution(sims: List[float], k: int, floor: float) -> Dict:
    """Summarize a query's candidate similarities (already the RPC's output)."""
    sims = sorted(sims, reverse=True)
    n = len(sims)
    return {
        "n_candidates": n,
        "sim_max": sims[0] if n else None,
        "sim_median": st.median(sims) if n else None,
        "sim_min": sims[-1] if n else None,
        # The k-th best is the weakest chunk retrieval would actually return at k.
        "kth_similarity": sims[min(k, n) - 1] if n else None,
        "cleared_floor": sum(1 for s in sims if s >= floor),
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "-"


def main() -> None:
    load_dotenv()
    config = AppConfig.from_env()
    if not config.supabase.enabled:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    embeddings = ServiceContainer(config).get_embedding_model().get_embeddings()
    client = create_client(config.supabase.url, config.supabase.service_role_key)

    rc = config.retrieval
    total_chunks = _count(client, "documents")
    total_articles = _count(client, "article_content")

    panel = (
        [(q, "fintech") for q in FINTECH_QUERIES]
        + [(q, "control") for q in CONTROL_QUERIES]
    )
    rows: List[Dict] = []
    for query, kind in panel:
        qv = embeddings.embed_query(query)
        params = {"query_embedding": qv, "match_count": rc.fetch_k}
        # Mirror retrieval's recency window so the probe sees the same candidate
        # pool the live retriever would (window_days=0 -> whole corpus).
        if rc.window_days and rc.window_days > 0:
            params["window_days"] = rc.window_days
        data = client.rpc("match_documents", params).execute().data or []
        sims = [float(r["similarity"]) for r in data if r.get("similarity") is not None]
        dist = _distribution(sims, rc.k, rc.min_similarity)
        rows.append(
            {
                "total_chunks": total_chunks,
                "total_articles": total_articles,
                "query": query,
                "query_kind": kind,
                "reference_floor": rc.min_similarity,
                **dist,
            }
        )
        logger.info(
            "%-8s %-42s n=%2d max=%s kth=%s cleared=%d",
            kind, query, dist["n_candidates"],
            _fmt(dist["sim_max"]), _fmt(dist["kth_similarity"]), dist["cleared_floor"],
        )

    client.table("corpus_probe").insert(rows).execute()

    # Today's grounded read: the separation between on-topic and off-topic.
    fin_floors = [r["kth_similarity"] for r in rows
                  if r["query_kind"] == "fintech" and r["kth_similarity"] is not None]
    ctrl_ceils = [r["sim_max"] for r in rows
                  if r["query_kind"] == "control" and r["sim_max"] is not None]
    print(f"\ncorpus: {total_chunks} chunks / {total_articles} articles  "
          f"(reference floor {rc.min_similarity})")
    if fin_floors and ctrl_ceils:
        ontopic_floor = min(fin_floors)
        offtopic_ceiling = max(ctrl_ceils)
        band = ontopic_floor - offtopic_ceiling
        print(f"on-topic floor (min fintech kth)    = {ontopic_floor:.3f}")
        print(f"off-topic ceiling (max control max) = {offtopic_ceiling:.3f}")
        print(f"band = {band:+.3f}   suggested floor today ~ "
              f"{(ontopic_floor + offtopic_ceiling) / 2:.3f}")
        if band <= 0:
            print("WARNING: bands overlap today - off-topic scored >= weakest "
                  "on-topic. Floor cannot cleanly separate; corpus needs breadth.")
    print(f"Inserted {len(rows)} probe rows into corpus_probe.")


if __name__ == "__main__":
    main()