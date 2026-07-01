-- Gold side-table: fintech articles that matched no theme.
--
-- When Gold aggregates trends, an article whose full text matches none of the
-- current themes would otherwise be dropped. Instead it lands here, so the
-- taxonomy's blind spots accumulate and can be analysed later to discover
-- themes worth adding. Not used by any trend computation - purely a capture log.
--
-- `text` is the full scraped article (same text the theme keywords were matched
-- against in Silver), so the gap analysis works from the real content.
--
-- Deduped by URL. Run in the Supabase SQL editor. Server-side only (RLS on).

create table if not exists untagged_articles (
    url           text        not null primary key,
    title         text        not null,
    text          text        not null default '',
    published_at  timestamptz not null,
    recorded_at   timestamptz not null default now()
);

alter table untagged_articles enable row level security;

-- Migration for an existing table (created when this column held the RSS
-- summary, before Gold read full text from Silver):
--   alter table untagged_articles rename column summary to text;
