-- Gold layer: per-category weekly trend metrics, across all three tag dimensions.
--
-- Aggregates the fintech (accepted) articles by ISO week and tag category,
-- giving coverage volume over time - the trend signal. Generalized beyond
-- themes: `dimension` is "theme", "risk", or "signal", and `category` is the
-- specific label within it. Recomputed from Silver each run and upserted, so
-- counts always reflect current data.
--
-- week_start is the Monday of the article's publish week. An article counts
-- toward every category it carries, in every dimension.
--
-- Run in the Supabase SQL editor. Server-side only via service_role (RLS on).

create table if not exists trend_metrics (
    week_start     date    not null,
    dimension      text    not null,
    category       text    not null,
    article_count  integer not null,
    primary key (week_start, dimension, category)
);

alter table trend_metrics enable row level security;

-- Migration from the theme-only schema (PK was (week_start, theme)). The table
-- is recomputed from Silver every Gold run, so the simplest path is to drop and
-- recreate it with the statement above:
--   drop table if exists trend_metrics;
