-- Gold layer: how common each tag category is across the whole fintech corpus.
--
-- The denominator behind lift-ranked thesis tags. A thesis ranks the tags its
-- retrieved articles carry; ranking those by raw count answers "which tag is
-- most common here?", and since the evidence pool is a large slice of the
-- corpus, the answer is usually just "whichever tag is most common everywhere".
-- Dividing the pool's rate by the rate stored here answers the question the UI
-- actually displays: "which tag is over-represented in THIS query's evidence?".
--
-- Same (dimension, category) vocabulary as trend_metrics, but aggregated over
-- the whole corpus rather than bucketed by week. `total_articles` is the shared
-- denominator (every row of one recompute carries the same value); it is stored
-- per row so a rate is readable without a second query.
--
-- Recomputed from Silver each Gold run and upserted, then any row this run did
-- NOT stamp is deleted - a category that left the corpus must not linger as a
-- live lift denominator.
--
-- Run in the Supabase SQL editor. Server-side only via service_role (RLS on).

create table if not exists tag_base_rates (
    dimension       text    not null,
    category        text    not null,
    article_count   integer not null,
    total_articles  integer not null,
    -- NOT NULL because the stale-row cleanup is `delete ... where computed_at
    -- <> <this run>`, and in Postgres NULL <> value is NULL, not true. An
    -- unstamped row would survive every recompute and stay a live lift
    -- denominator forever.
    computed_at     timestamptz not null,

    primary key (dimension, category),
    constraint tag_base_rates_count_nonneg check (article_count >= 0),
    constraint tag_base_rates_count_within_total check (article_count <= total_articles)
);

alter table tag_base_rates enable row level security;

-- Migration for an environment that already created this table without the
-- NOT NULL (every row Gold writes is stamped, so this should not rewrite any):
--   update tag_base_rates set computed_at = now() where computed_at is null;
--   alter table tag_base_rates alter column computed_at set not null;
