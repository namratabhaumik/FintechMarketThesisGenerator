-- Silver layer: one classification verdict per processed Bronze article.
--
-- Records EVERY article the Silver build has decided on - both fintech
-- (embedded into `documents`) and rejected (never embedded) - so a later run
-- skips them and never re-runs classification. This is separate from the
-- vector store, which holds only the accepted/embedded subset.
--
-- Run in the Supabase SQL editor. Server-side only via service_role (RLS on).

create table if not exists articles_silver (
    url               text        not null primary key,
    fintech_relevant  boolean     not null,
    -- Three deterministic tag dimensions matched on the full text at Silver
    -- time. Gold accumulates each into historic trends; the thesis reads them
    -- (via the embedded chunk metadata) to stay grounded.
    themes            jsonb       not null default '[]'::jsonb,
    risks             jsonb       not null default '[]'::jsonb,
    signals           jsonb       not null default '[]'::jsonb,
    processed_at      timestamptz not null default now(),
    load_id           uuid
);

alter table articles_silver enable row level security;

-- Migration for an existing table:
--   alter table articles_silver add column if not exists load_id uuid;

-- Migration for an existing table (created before themes existed):
--   alter table articles_silver add column if not exists themes jsonb not null default '[]'::jsonb;
-- Migration for the risk/signal dimensions (added when tagging moved fully to Silver):
--   alter table articles_silver add column if not exists risks jsonb not null default '[]'::jsonb;
--   alter table articles_silver add column if not exists signals jsonb not null default '[]'::jsonb;
-- Existing rows get []; clear them so Silver recomputes all three on the full text:
--   delete from articles_silver;
