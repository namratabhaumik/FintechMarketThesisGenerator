-- Create the jobs table for thesis generation job tracking.
-- Run this in your Supabase SQL Editor (Dashboard > SQL Editor > New Query).

create extension if not exists vector;

create table if not exists jobs (
    id           text primary key,
    query        text not null,
    status       text not null default 'pending',
    progress     text,
    thesis       jsonb,
    articles     jsonb not null default '[]'::jsonb,
    error        text,
    refinement_count    integer not null default 0,
    refinement_status   text not null default 'refining',
    feedback_history    jsonb not null default '[]'::jsonb,
    execution_log       jsonb not null default '[]'::jsonb,
    retrieved_docs      jsonb not null default '[]'::jsonb,
    -- Prior thesis versions (one per completed refinement round), so the UI's
    -- "Previous versions" history survives a refresh/resume, paired with
    -- feedback_history. Serialized StructuredThesis dicts, oldest first.
    thesis_history      jsonb not null default '[]'::jsonb,
    -- Approval timestamp (episodic memory): NULL = not approved, else the
    -- time a human approved the thesis.
    approved_at         timestamptz,
    -- Query embedding (episodic recall): the run's query vector (FastEmbed jina,
    -- 512-dim) stored as pgvector, used to rank past runs by cosine similarity
    -- via match_jobs. Older rows are NULL and are skipped by recall.
    query_embedding     vector(512),
    -- Owner (multi-tenant isolation). Set to the authenticated user on insert
    -- via the per-request user-scoped client (default auth.uid()); RLS scopes
    -- all access to auth.uid(). NULL only on legacy/pre-auth rows (invisible
    -- under RLS).
    user_id      uuid references auth.users(id) on delete cascade default auth.uid(),
    created_at   timestamptz not null default now()
);

-- HNSW index for fast cosine-similarity recall over query embeddings.
create index if not exists jobs_query_embedding_hnsw
    on jobs using hnsw (query_embedding vector_cosine_ops);

-- Row Level Security: each user reaches only their own jobs. The API's
-- per-request user-scoped client (anon key + the user's JWT) is subject to
-- these policies; the service_role key bypasses RLS (admin/maintenance only).
alter table jobs enable row level security;

create policy "jobs_select_own" on jobs
  for select using (auth.uid() = user_id);
create policy "jobs_insert_own" on jobs
  for insert with check (auth.uid() = user_id);
create policy "jobs_update_own" on jobs
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "jobs_delete_own" on jobs
  for delete using (auth.uid() = user_id);

-- RLS filters every query by user_id; index it.
create index if not exists jobs_user_id_idx on jobs (user_id);

-- match_jobs: past runs most similar to query_embedding (episodic recall),
-- excluding the current run and runs without a thesis, above min_similarity.
-- Mirrors match_documents but ranks jobs by their query embedding.
drop function if exists match_jobs(vector, text, int, float);

create function match_jobs (
  query_vec vector(512),
  exclude_id text,
  match_count int default 3,
  min_similarity float default 0.86
) returns table (
  job_id text,
  query text,
  created_at timestamptz,
  score float,
  recommendation text,
  approved boolean,
  similarity float
)
language sql stable
as $$
  select
    jobs.id,
    jobs.query,
    jobs.created_at,
    (jobs.thesis->>'opportunity_score')::float,
    jobs.thesis->>'recommendation',
    jobs.approved_at is not null,
    round((1 - (jobs.query_embedding <=> query_vec))::numeric, 2)::float
  from jobs
  where jobs.id <> exclude_id
    and jobs.thesis is not null
    and jobs.query_embedding is not null
    and (jobs.query_embedding <=> query_vec) <= 1 - min_similarity
  order by jobs.query_embedding <=> query_vec
  limit match_count;
$$;

-- Migration for an existing table whose query_embedding is jsonb (pre-pgvector).
--   Step 1 (keeping the jsonb aside for rollback):
--     create extension if not exists vector;
--     alter table jobs rename column query_embedding to query_embedding_old;
--     alter table jobs add column query_embedding vector(512);
--     update jobs set query_embedding = query_embedding_old::text::vector
--       where query_embedding_old is not null;
--     create index if not exists jobs_query_embedding_hnsw
--       on jobs using hnsw (query_embedding vector_cosine_ops);
--     -- then create match_jobs (above)
--   Step 2 (after testing confirm recall works):
--     alter table jobs drop column query_embedding_old;
--
-- Older migration notes:
--   alter table jobs add column if not exists approved_at timestamptz;
--   alter table jobs add column if not exists thesis_history jsonb not null default '[]'::jsonb;
-- Migration for a table created with the earlier allow-all anon policy:
--   drop policy if exists "Allow all access via anon key" on jobs;
--
-- Multi-tenant migration for an existing jobs table (add owner + RLS policies):
--   alter table jobs add column if not exists user_id uuid
--     references auth.users(id) on delete cascade default auth.uid();
--   create index if not exists jobs_user_id_idx on jobs (user_id);
--   -- then create the four jobs_*_own policies above.
--   -- Existing rows have user_id = NULL and are invisible under RLS; either
--   -- backfill them (update jobs set user_id = '<uuid>' ...) or delete test rows.
