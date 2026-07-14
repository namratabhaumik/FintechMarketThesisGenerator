-- documents: the pgvector store of embedded article chunks (Silver write path,
-- retrieval read path). Holds only the accepted/fintech subset, one row per
-- chunk, with the source article's tags + publish date in `metadata`.
--
-- This is the LangChain SupabaseVectorStore schema plus a custom
-- `match_documents` that adds a recency filter (`window_days`) on
-- metadata->>'published_at'. The retrieval layer calls match_documents to pull
-- `match_count` nearest chunks by cosine distance, then does MMR in Python.
--
-- DIMENSION: `embedding` is vector(512) for jinaai/jina-embeddings-v2-small-en.
-- The embedding column and BOTH match_documents vector types must agree with
-- EMBEDDING_MODEL's output dim. Switching models = change the dim in all three
-- places here, then re-embed the corpus (scripts/reembed_documents.py), because
-- pgvector cannot mix dimensions in one column.
--
-- RLS: the app connects only with the service-role key, which BYPASSES RLS, so
-- enabling RLS with no policies is safe and is the secure default (it blocks
-- anon/public access while leaving the backend unaffected). Add policies only
-- if some client ever reads this table with the anon key.
--
-- Run in the Supabase SQL editor.

create extension if not exists vector;

-- Recreate at 512 dims. `drop ... cascade` discards the old 384-dim chunks
-- (incompatible) and any dependents; re-embed repopulates from article_content.
drop table if exists documents cascade;
create table documents (
    id uuid primary key default gen_random_uuid(),
    content text,
    metadata jsonb,
    embedding vector(512),
    created_at timestamptz default now()
);
create index on documents using hnsw (embedding vector_cosine_ops);

-- match_documents: nearest chunks by cosine distance, optionally restricted to
-- the last `window_days` OR an explicit `date_from`/`date_to` range (used
-- when the query names one, e.g. "since March 2024" - see
-- core/utils/date_intent.py; the retrieval service sets only one of
-- window_days or date_from/date_to per call, never both). Drop EVERY
-- plausible overload first: Postgres treats a different arg-type order as a
-- separate function, so a single drop can leave a stale overload behind and
-- make PostgREST RPC resolution ambiguous. These cover the canonical order,
-- the (window_days, filter)-swapped order, the 3-arg legacy form, and the
-- prior 4-arg form (before date_from/date_to were added).
drop function if exists match_documents(vector, int, jsonb, int);
drop function if exists match_documents(vector, int, int, jsonb);
drop function if exists match_documents(vector, int, jsonb);
drop function if exists match_documents(vector, int, jsonb, int, timestamptz, timestamptz);

create function match_documents (
  query_embedding vector(512),
  match_count int default null,
  filter jsonb default '{}',
  window_days int default null,
  date_from timestamptz default null,
  date_to timestamptz default null
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  embedding vector(512),
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    documents.embedding,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
    and (
      window_days is null
      or (documents.metadata->>'published_at')::timestamptz >= now() - make_interval(days => window_days)
    )
    and (
      date_from is null
      or (documents.metadata->>'published_at')::timestamptz >= date_from
    )
    and (
      date_to is null
      or (documents.metadata->>'published_at')::timestamptz <= date_to
    )
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
