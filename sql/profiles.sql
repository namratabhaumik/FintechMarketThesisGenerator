-- profiles: display identity for annotation authors.
--
-- Deliberately NOT a public directory: a row is readable by its owner and by an
-- admin, which is all the annotation panel needs while a thesis is owner-only.
-- Widening this is part of sharing.
--
-- display_name is a name, not an email.

create table if not exists profiles (
    id           uuid primary key references auth.users(id) on delete cascade,
    display_name text,
    avatar_url   text,
    updated_at   timestamptz not null default now()
);

alter table profiles enable row level security;

-- Own row, plus admin (who can open any thesis and so any thread on it).
drop policy if exists "profiles_select_authenticated" on profiles;
drop policy if exists "profiles_select_own" on profiles;
create policy "profiles_select_own" on profiles
  for select using (
    auth.uid() = id
    or (auth.jwt() -> 'app_metadata' ->> 'role') = 'admin'
  );

-- A user may only ever write their own row. Inserts come from the trigger
-- below (which runs as definer), so this only governs self-service edits.
create policy "profiles_update_own" on profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

-- Populate on signup. SECURITY DEFINER because the trigger runs in auth's
-- context, not the new user's, and must bypass the policies above.
-- The metadata keys are what Google's OAuth provider returns; both are left
-- null for providers that don't supply them (no email fallback - see the header).
create or replace function handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into profiles (id, display_name, avatar_url)
  values (
    new.id,
    coalesce(
      new.raw_user_meta_data ->> 'full_name',
      new.raw_user_meta_data ->> 'name'
    ),
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function handle_new_user();

-- Backfill for accounts that already existed when this landed: the trigger only
-- fires on new signups, so without this every current user is nameless. Safe to
-- re-run (on conflict do nothing).
insert into profiles (id, display_name, avatar_url)
select
  u.id,
  coalesce(
    u.raw_user_meta_data ->> 'full_name',
    u.raw_user_meta_data ->> 'name'
  ),
  u.raw_user_meta_data ->> 'avatar_url'
from auth.users u
on conflict (id) do nothing;

-- Clean-up for environments that ran the earlier version of this file, which
-- fell back to the email address. `on conflict do nothing` above will not
-- correct rows that already exist, so clear the ones holding an email.
update profiles p
   set display_name = null
  from auth.users u
 where u.id = p.id
   and p.display_name = u.email;
