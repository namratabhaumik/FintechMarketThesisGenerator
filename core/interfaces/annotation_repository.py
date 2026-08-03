"""Abstract interface for annotation and profile persistence."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class IAnnotationRepository(ABC):
    """Protocol for collaborative annotation storage.

    Covers one bounded context: the notes on a thesis and the display
    identities of the people who wrote them. They are grouped because neither is
    useful alone - rendering a thread needs both - and because they answer to the
    same access rules.

    Implementations decide how/where this is persisted and how access is
    enforced. Routes depend on this abstraction so the backend can be swapped
    without touching API code.

    Two conventions the routes rely on, which implementations must honour:

    - Reads and deletes are access-scoped, not error-signalling. A read that the
      caller may not see returns None (or an empty list), and a delete they may
      not perform is a silent no-op. Callers translate that into 404/403; an
      implementation must not raise merely because access was denied.
    - Authorship is never taken from the caller. `create_*` must derive the
      author from the authenticated session, not from arguments, so a client
      cannot post as someone else.
    """

    # --- Annotations ---

    @abstractmethod
    async def list_annotations(self, job_id: str, version: Optional[int] = None) -> list[dict]:
        """Annotations on a thesis, oldest first, roots and replies together.

        Returning both in one call lets the caller thread by parent_id without a
        second round trip. `version` scopes to one version's anchors; omitting it
        returns every version.
        """
        pass

    @abstractmethod
    async def get_annotation(self, annotation_id: str) -> Optional[dict]:
        """One annotation, or None if it does not exist or is not visible."""
        pass

    @abstractmethod
    async def create_annotation(self, **fields: Any) -> Optional[dict]:
        """Insert an annotation and return the stored row.

        Returns None when the caller is not permitted to annotate this thesis.
        """
        pass

    @abstractmethod
    async def update_annotation_body(self, annotation_id: str, body: str) -> Optional[dict]:
        """Edit a comment's text, returning the updated row.

        Author-only: returns None for anyone else. Editing must never be
        widened beyond the author, since resolution exists for the state
        changes other participants are allowed to make.
        """
        pass

    @abstractmethod
    async def delete_annotation(self, annotation_id: str) -> None:
        """Delete an annotation; its replies go with it."""
        pass

    @abstractmethod
    async def set_resolution(self, annotation_id: str, resolution: Optional[str]) -> None:
        """Tick ('accepted') or reopen (None) a thread.

        Distinct from update_annotation_body: resolution is thread state, not
        anyone's words. Unlike the access-scoped reads above, this raises when
        the annotation is missing or the caller may not resolve it - the caller
        cannot otherwise tell a forbidden resolve from a successful one.
        """
        pass

    # --- Profiles ---

    @abstractmethod
    async def get_profiles(self, user_ids: list[str]) -> dict[str, dict]:
        """Display identities for a set of authors, keyed by user id.

        Batched deliberately: threads render many authors, and a per-row lookup
        would be N+1. Unknown ids are omitted rather than raising - a profile can
        lag a new signup, and a nameless author must not break the thread.
        """
        pass
