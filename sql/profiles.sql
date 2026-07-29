-- profiles: public, readable identity for annotation authors.

create table if not exists profiles (
    id           uuid primary key references auth.users(id) on delete cascade,
    display_name text,
    avatar_url   text,
    updated_at   timestamptz not null default now()
);

alter table profiles enable row level security;

-- Any signed-in user may read any profile: annotation threads are shared, so
-- rendering one means rendering other people's names.
create policy "profiles_select_authenticated" on profiles
  for select using (auth.role() = 'authenticated');

-- A user may only ever write their own row. Inserts come from the trigger
-- below (which runs as definer), so this only governs self-service edits.
create policy "profiles_update_own" on profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

-- Populate on signup. SECURITY DEFINER because the trigger runs in auth's
-- context, not the new user's, and must bypass the policies above.
-- The metadata keys are what Google's OAuth provider returns; both are left
-- null for providers that don't supply them.
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
      new.raw_user_meta_data ->> 'name',
      new.raw_user_meta_data ->> 'email'
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
    u.raw_user_meta_data ->> 'name',
    u.raw_user_meta_data ->> 'email'
  ),
  u.raw_user_meta_data ->> 'avatar_url'
from auth.users u
on conflict (id) do nothing;