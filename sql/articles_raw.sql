-- Bronze layer: raw RSS feed entries, landed verbatim.
--
-- One row per feed entry (title, summary, link, publish date) with no
-- full-text scrape, classification or embedding - those happen in Silver,
-- which reads these rows. Append-only and deduped by URL so the corpus
-- accumulates over time on the published_at axis.
--
-- Run in the Supabase SQL editor. Accessed server-side with the service_role
-- key, which bypasses RLS.

create table if not exists articles_raw (
    id            bigint generated always as identity primary key,
    url           text        not null unique,
    feed_name     text        not null default '',
    source        text        not null default '',
    title         text        not null,
    summary       text        not null default '',
    published_at  timestamptz not null,
    fetched_at    timestamptz not null default now()
);

-- Trend aggregation reads by publish time; newest-first is the common scan.
create index if not exists articles_raw_published_at_idx
    on articles_raw (published_at desc);
