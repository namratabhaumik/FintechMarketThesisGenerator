-- corpus_probe: a daily read-only snapshot of the retrieval layer's health,
-- written by scripts.inspect_corpus (wired into the daily-ingest workflow).
--
-- One row per (run, query). ~10 queries/day, ~300 rows over a 30-day window.
--
-- RLS: the app connects only with the service-role key, which BYPASSES RLS, so
-- enabling RLS with no policies blocks anon/public access while leaving the
-- backend (and this script) unaffected. Same as the other tables.
--
-- Run in the Supabase SQL editor.

create table if not exists corpus_probe (
    id uuid primary key default gen_random_uuid(),
    run_date date not null default (now() at time zone 'utc')::date,
    -- Corpus size at probe time (denormalized onto every query row for the run).
    total_chunks int not null,
    total_articles int not null,
    -- The probed query and whether it is an on-topic fintech query or an
    -- off-topic control (the two are read differently: fintech -> floor,
    -- control -> ceiling).
    query text not null,
    query_kind text not null check (query_kind in ('fintech', 'control')),
    -- Similarity distribution of the fetch_k candidates match_documents returned.
    -- Nullable: a query can return zero candidates on a tiny/empty corpus.
    n_candidates int not null,
    sim_max float,
    sim_median float,
    sim_min float,
    -- kth_similarity: the similarity of the k-th candidate, i.e. the weakest
    -- chunk retrieval would actually RETURN at k. For fintech queries this is the
    -- "on-topic floor" the min_similarity threshold must stay below.
    kth_similarity float,
    -- The reference floor in effect at probe time (RetrievalConfig.min_similarity)
    -- and how many candidates cleared it. Per query, shows whether the current
    -- floor is admitting a healthy number of chunks or starving the query.
    reference_floor float not null,
    cleared_floor int not null,
    created_at timestamptz not null default now()
);

create index if not exists corpus_probe_run_date_idx on corpus_probe (run_date);

alter table corpus_probe enable row level security;