-- Create the jobs table for thesis generation job tracking.
-- Run this in your Supabase SQL Editor (Dashboard > SQL Editor > New Query).

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
    -- Query embedding (episodic recall): the run's query vector (FastEmbed) as a
    -- JSON array of floats, used to rank past runs by cosine similarity. Older
    -- rows are NULL and are skipped by recall until regenerated.
    query_embedding     jsonb,
    created_at   timestamptz not null default now()
);

-- Row Level Security: on with no policy, so only the service_role key (which
-- bypasses RLS) can reach this table. The app connects with service_role.
alter table jobs enable row level security;

-- Migration for an existing table (created before approval/recall columns existed):
--   alter table jobs add column if not exists approved_at timestamptz;
--   alter table jobs add column if not exists query_embedding jsonb;
--   alter table jobs add column if not exists thesis_history jsonb not null default '[]'::jsonb;
-- Migration for a table created with the earlier allow-all anon policy:
--   drop policy if exists "Allow all access via anon key" on jobs;
