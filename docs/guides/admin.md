# Administration

Accounts with the `admin` role get a cross-user management view in addition to their own personal library. Regular accounts never see it, and the API refuses the underlying requests for them (`403`), so the boundary holds even outside the UI.

<!-- Clip: assets/clips/admin.mp4
<video controls muted playsinline width="100%">
  <source src="../assets/clips/admin.mp4" type="video/mp4">
</video>
-->

## The all-users view

Below the personal **Past theses** library, admins see a second, clearly separated section listing every other user's theses (the admin's own rows stay in the personal library above, so nothing appears twice). Each row carries the same summary fields as the personal library plus an `owner` label with a short prefix of the owning account's id.

Rows are clickable like any library entry: an admin can open any user's thesis in full, including its versions and execution trace.

## Deleting a thesis

Each row in the all-users view has a delete control. Deletion asks for confirmation (quoting the thesis query), is admin-only, and is permanent - there is no trash or undo. It removes the job and all its state: versions, feedback history, execution log.

## How the role is assigned

The admin role lives in the Supabase account's app metadata (`app_metadata.role = "admin"`), set by the operator in Supabase - it cannot be self-assigned from the app. The frontend reads it only to decide whether to render the admin section; authorization is enforced server-side on every request, backed by row-level security policies that grant admins cross-user reads.
