-- Silver record: the validated, non-aggregated representation of each article.
--
-- The full scraped+cleaned article text, stored once it passes Article
-- validation. This is the durable source the embedding step transforms from:
-- documents (pgvector) holds the chunked/aggregated embeddings, article_content
-- holds the one clean record per URL.
--
-- Persisting the text decouples embedding from the costly, non-reproducible
-- scrape: a re-embed (or an embed retry after a failed run) reads the text from
-- here instead of scraping again. Scrape happens once per URL.
--
-- Run in the Supabase SQL editor. Server-side only (RLS on).

create table if not exists article_content (
    url           text        not null primary key,
    title         text        not null,
    text          text        not null,
    source        text        not null,
    published_at  timestamptz not null,
    created_at    timestamptz not null default now(),
    load_id       uuid
);

alter table article_content enable row level security;

-- Migration for an existing table:
--   alter table article_content add column if not exists load_id uuid;
