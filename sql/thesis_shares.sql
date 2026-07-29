-- thesis_shares: who, besides the owner, may see and annotate a thesis.
--
-- Annotations are collaborative, so access can no longer be "auth.uid() =
-- user_id" everywhere - a second user has to be able to read a thesis and add
-- notes to it without owning it. This table is that grant, and every
-- collaborator-aware policy (see annotations.sql) resolves through it.
--
-- Deliberately narrow: a share conveys read + comment. It never conveys
-- refine, approve or delete, which stay with the owner (and admins).

create table if not exists thesis_shares (
    job_id     text        not null references jobs(id) on delete cascade,
    user_id    uuid        not null references auth.users(id) on delete cascade,
    -- 'commenter' may add annotations; 'viewer' may only read them.
    role       text        not null default 'commenter',
    -- Who granted it, for auditability.
    granted_by uuid        references auth.users(id) on delete set null,
    created_at timestamptz not null default now(),
    primary key (job_id, user_id),
    constraint thesis_shares_role_valid check (role in ('commenter', 'viewer'))
);

-- Policies resolve shares by job (does this job grant me access?) and by user
-- (which theses am I a collaborator on?); index both directions.
create index if not exists thesis_shares_user_id_idx on thesis_shares (user_id);

alter table thesis_shares enable row level security;

-- You can see a share if it is yours, or if you own the thesis it grants.
create policy "thesis_shares_select" on thesis_shares
  for select using (
    auth.uid() = user_id
    or exists (
      select 1 from jobs j
      where j.id = thesis_shares.job_id and j.user_id = auth.uid()
    )
  );

-- Only the thesis owner may grant or revoke access to it. Sharing is not
-- transitive: a collaborator cannot re-share someone else's thesis.
create policy "thesis_shares_insert_owner" on thesis_shares
  for insert with check (
    exists (
      select 1 from jobs j
      where j.id = thesis_shares.job_id and j.user_id = auth.uid()
    )
  );

create policy "thesis_shares_delete_owner" on thesis_shares
  for delete using (
    exists (
      select 1 from jobs j
      where j.id = thesis_shares.job_id and j.user_id = auth.uid()
    )
  );

-- Admins manage shares across users, mirroring the admin policies on jobs.
create policy "thesis_shares_admin_all" on thesis_shares
  for all using ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin')
  with check ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin');
