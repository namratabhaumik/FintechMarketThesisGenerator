-- Silver dead-letter: articles that failed enrichment.
--
-- When Silver cannot produce a clean embedded record - the full-text scrape
-- returned nothing, or the Article failed validation - the URL is parked here
-- instead of embedding a thin fallback or being retried every run. Silver
-- excludes quarantined URLs from processing, so they stay out of the corpus
-- until replayed.
--
-- Replay = delete the row: the next Silver run re-attempts the URL (still in
-- articles_raw). Bronze remains the source of truth, so only inspection fields
-- are stored here.
--
-- Run in the Supabase SQL editor. Server-side only (RLS on).

create table if not exists quarantine (
    url          text        not null primary key,
    reason       text        not null,  -- scrape_failed | invalid_article
    detail       text        not null default '',
    title        text        not null default '',
    recorded_at  timestamptz not null default now()
);

alter table quarantine enable row level security;
