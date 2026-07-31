"""Supabase-backed store for annotations and author profiles.

Every method runs on a user-scoped client (anon key + the caller's JWT), so RLS
decides what is visible or writable - see sql/annotations.sql. Nothing here
re-implements those checks; a query that returns no rows means the policies said
no, which the routes translate into 404/403 as appropriate.

The one exception is resolution: RLS is row-level, so a policy permissive
enough to set `resolution` would equally permit rewriting that annotation's
`body`. That path goes through the set_annotation_resolution definer function
instead, which touches only the state columns.
"""

import logging
from typing import Any, Optional

from supabase import AsyncClient

from core.interfaces.annotation_repository import IAnnotationRepository

logger = logging.getLogger(__name__)

ANNOTATIONS = "annotations"
PROFILES = "profiles"


class SupabaseAnnotationManager(IAnnotationRepository):
    """Annotation and profile reads and writes for one request's caller.

    The interface's access-scoped contract falls out of RLS here: a policy that
    denies the caller matches no rows, so reads come back empty and writes
    return None without raising.
    """

    def __init__(self, client: AsyncClient):
        self._client = client

    # --- Annotations ---

    async def list_annotations(
        self, job_id: str, version: Optional[int] = None
    ) -> list[dict]:
        """Every annotation on a thesis, oldest first.

        Replies are returned alongside roots (the caller threads them by
        parent_id) so a thread never needs a second round trip. `version` scopes
        to one version's anchors; omitting it returns all versions, which the
        panel uses to show that earlier versions carry notes.
        """
        query = self._client.table(ANNOTATIONS).select("*").eq("job_id", job_id)
        if version is not None:
            query = query.eq("version", version)
        resp = await query.order("created_at", desc=False).execute()
        return resp.data or []

    async def get_annotation(self, annotation_id: str) -> Optional[dict]:
        resp = (
            await self._client.table(ANNOTATIONS)
            .select("*")
            .eq("id", annotation_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None

    async def create_annotation(self, **fields: Any) -> Optional[dict]:
        """Insert an annotation. user_id defaults to auth.uid() in the DB, so it
        is never taken from the client."""
        resp = await self._client.table(ANNOTATIONS).insert(fields).execute()
        rows = resp.data or []
        return rows[0] if rows else None

    async def update_annotation_body(
        self, annotation_id: str, body: str
    ) -> Optional[dict]:
        """Edit a comment. RLS restricts this to the author's own rows, so a
        non-author's update simply matches nothing."""
        resp = (
            await self._client.table(ANNOTATIONS)
            .update({"body": body})
            .eq("id", annotation_id)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None

    async def delete_annotation(self, annotation_id: str) -> None:
        """Delete an annotation (replies cascade). RLS scopes this to the
        author or the thesis owner; anything else is a silent no-op."""
        await self._client.table(ANNOTATIONS).delete().eq("id", annotation_id).execute()

    async def set_resolution(
        self, annotation_id: str, resolution: Optional[str]
    ) -> None:
        """Tick or reopen a thread via the definer function.

        Raises whatever the function raises (not found / not authorised); the
        route maps that onto a status code.
        """
        await self._client.rpc(
            "set_annotation_resolution",
            {"annotation_id": annotation_id, "new_resolution": resolution},
        ).execute()

    # --- Profiles ---

    async def get_profiles(self, user_ids: list[str]) -> dict[str, dict]:
        """Display names/avatars for a set of authors, keyed by user id.

        Fetched in one query per annotation list rather than per row. Missing
        ids are simply absent - a profile row can lag a brand-new signup, and a
        nameless author must not break the thread.
        """
        if not user_ids:
            return {}
        resp = (
            await self._client.table(PROFILES)
            .select("id, display_name, avatar_url")
            .in_("id", list(set(user_ids)))
            .execute()
        )
        return {row["id"]: row for row in (resp.data or [])}
