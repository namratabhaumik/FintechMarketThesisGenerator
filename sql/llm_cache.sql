-- llm_cache: persistent store for AI Gateway LLM responses.
--
-- Backs SupabaseCacheManager (AI_GATEWAY_CACHE_TYPE=supabase). Unlike the
-- in-memory CacheManager, this survives process restarts / Render cold starts
-- and is shared across instances. `key` is a SHA256 of
--   <cache_version> | <documents_text> | <topic> | <discriminator>
-- so a bump of AI_GATEWAY_CACHE_VERSION (or a model change, which the container
-- folds into the version) busts every entry. TTL is enforced in the app on
-- read; created_at supports that and any out-of-band cleanup.

create table if not exists llm_cache (
    key           text primary key,
    response      text        not null,
    model         text        not null,
    input_tokens  int         not null default 0,
    output_tokens int         not null default 0,
    created_at    timestamptz not null default now()
);

-- Age lookups for TTL eviction / cleanup by recency.
create index if not exists llm_cache_created_at_idx on llm_cache (created_at);
