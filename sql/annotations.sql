-- annotations: highlights and threaded comments on a thesis.
--
-- VERSION PINNING: an annotation is anchored to one specific version of a thesis, 
-- never migrated forward. Once a version is superseded its text is frozen in 
-- jobs.thesis_history, so the offsets can never drift under the note. Refining 
-- produces a new version with no annotations.
--
-- `version` is 1-based and matches what the UI labels: version n is
-- thesis_history[n-1] once superseded, and jobs.thesis while current.
--
-- THREADING: one table, self-referencing. A root row carries the anchor (which
-- passage it marks); a reply carries only a body and its parent. The check
-- constraint keeps those two shapes from mixing - a reply cannot smuggle in its
-- own anchor, and a root cannot exist without one.

create table if not exists annotations (
    id           uuid        primary key default gen_random_uuid(),
    job_id       text        not null references jobs(id) on delete cascade,
    version      int         not null,

    -- Anchor (root rows only). `section` names which block of the thesis the
    -- offsets index into, so a note on Risks can never resolve against the
    -- Raw Summary. Offsets are character positions in that section's plain
    -- text; `quote` is the anchored text itself, stored because the panel
    -- displays it and because it makes a mis-anchor visible instead of silent.
    section      text,
    start_offset int,
    end_offset   int,
    quote        text,

    -- Reply target. Null for a root annotation.
    parent_id    uuid        references annotations(id) on delete cascade,

    body         text        not null,

    -- Resolution state (root rows only). Null = still open; 'accepted' is the
    -- tick, 'rejected' the cross. A rejection's reason is NOT stored here - it
    -- is posted as an ordinary reply authored by whoever rejected it, so every
    -- piece of prose in this table belongs to the person who wrote it.
    resolution   text,
    resolved_at  timestamptz,
    resolved_by  uuid        references auth.users(id) on delete set null,

    user_id      uuid        not null references auth.users(id) on delete cascade default auth.uid(),
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),

    constraint annotations_body_not_blank check (length(btrim(body)) > 0),
    constraint annotations_version_positive check (version > 0),
    constraint annotations_section_valid check (
        section is null
        or section in ('raw_summary', 'key_themes', 'risks', 'investment_signals')
    ),
    constraint annotations_range_valid check (
        start_offset is null or end_offset is null or end_offset > start_offset
    ),
    constraint annotations_resolution_valid check (
        resolution is null or resolution in ('accepted', 'rejected')
    ),
    -- Only a root annotation can be resolved; a reply has no state of its own.
    constraint annotations_resolution_root_only check (
        resolution is null or parent_id is null
    ),
    -- Root rows are anchored; replies never are.
    constraint annotations_anchor_shape check (
        (parent_id is null
            and section is not null
            and start_offset is not null
            and end_offset is not null
            and quote is not null)
        or
        (parent_id is not null
            and section is null
            and start_offset is null
            and end_offset is null
            and quote is null)
    )
);

-- The panel always loads by (job, version); replies always load by root.
create index if not exists annotations_job_version_idx on annotations (job_id, version);
create index if not exists annotations_parent_idx on annotations (parent_id);

-- Keep updated_at honest without relying on every writer to remember it (an
-- edited comment must show as edited even if changed via the SQL editor).
create or replace function set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists annotations_set_updated_at on annotations;
create trigger annotations_set_updated_at
  before update on annotations
  for each row execute function set_updated_at();

alter table annotations enable row level security;

-- Visibility follows the thesis, not the annotation: you may read notes on a
-- thesis you own, one shared with you, or (as an admin) any thesis. This is the
-- first policy shape here that is not a plain "auth.uid() = user_id" - that
-- form cannot express "someone else's row, on a thesis I can see".
create policy "annotations_select" on annotations
  for select using (
    exists (select 1 from jobs j where j.id = annotations.job_id and j.user_id = auth.uid())
    or exists (
      select 1 from thesis_shares s
      where s.job_id = annotations.job_id and s.user_id = auth.uid()
    )
    or (auth.jwt() -> 'app_metadata' ->> 'role') = 'admin'
  );

-- Writing requires authorship AND access: owner, or a collaborator whose share
-- is 'commenter' (a 'viewer' share reads but never writes).
create policy "annotations_insert" on annotations
  for insert with check (
    auth.uid() = user_id
    and (
      exists (select 1 from jobs j where j.id = annotations.job_id and j.user_id = auth.uid())
      or exists (
        select 1 from thesis_shares s
        where s.job_id = annotations.job_id
          and s.user_id = auth.uid()
          and s.role = 'commenter'
      )
    )
  );

-- Editing is author-only (nobody rewrites anyone else's words). Resolving is
-- deliberately NOT covered here - see set_annotation_resolution below for why
-- this policy cannot express it.
create policy "annotations_update_own" on annotations
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- Authors delete their own notes; thesis owners may also clear notes from their
-- own thesis (replies cascade via parent_id).
create policy "annotations_delete" on annotations
  for delete using (
    auth.uid() = user_id
    or exists (select 1 from jobs j where j.id = annotations.job_id and j.user_id = auth.uid())
  );

create policy "annotations_admin_all" on annotations
  for all using ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin')
  with check ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin');

-- Resolving a thread (tick = accepted, cross = rejected) is a state change any
-- participant may make on someone else's note. The update policy above cannot
-- express that (RLS is row-level, not column-level), so any policy permissive
-- enough to let a collaborator set `resolution` would equally let them rewrite
-- that annotation's `body`.
--
-- This function is the narrow exception. It re-checks thesis access itself and
-- writes ONLY the three state columns - `body` is never in its UPDATE, so no
-- caller can alter another person's words through it. A rejection's reason is
-- not passed here at all: the client posts it as an ordinary reply, authored by
-- and attributed to whoever rejected the note.
--
-- Pass null to reopen.
create or replace function set_annotation_resolution(annotation_id uuid, new_resolution text)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  target_job text;
begin
  if new_resolution is not null and new_resolution not in ('accepted', 'rejected') then
    -- 22023 = invalid_parameter_value.
    raise exception 'resolution must be accepted, rejected or null' using errcode = '22023';
  end if;

  -- Only root annotations carry resolution state; a reply has none to set.
  select job_id into target_job
  from annotations
  where id = annotation_id and parent_id is null;

  if target_job is null then
    -- Explicit SQLSTATEs, not prose: the API maps these to 404/403, and matching
    -- on message text would break silently the moment this wording changed.
    -- P0002 = no_data_found, 42501 = insufficient_privilege.
    raise exception 'annotation not found or is a reply' using errcode = 'P0002';
  end if;

  -- Resolving requires comment rights. A 'viewer' share
  -- is read-only and rejecting a thread is posting a reply with a reason.
  if not (
    exists (select 1 from jobs j where j.id = target_job and j.user_id = auth.uid())
    or exists (
      select 1 from thesis_shares s
      where s.job_id = target_job
        and s.user_id = auth.uid()
        and s.role = 'commenter'
    )
    or (auth.jwt() -> 'app_metadata' ->> 'role') = 'admin'
  ) then
    raise exception 'not authorised to resolve this annotation' using errcode = '42501';
  end if;

  update annotations
     set resolution  = new_resolution,
         resolved_at = case when new_resolution is null then null else now() end,
         resolved_by = case when new_resolution is null then null else auth.uid() end
   where id = annotation_id;
end;
$$;