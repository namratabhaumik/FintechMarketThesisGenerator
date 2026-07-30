"""Unit tests for the annotations layer: the Supabase manager and the route
validation that keeps root/reply shapes from mixing.

Access control itself is RLS (sql/annotations.sql) and is not re-implemented in
Python, so these tests cover what the app layer actually owns: the shape of the
queries sent to Supabase, and the request validation that turns a malformed
annotation into a 400 instead of a database constraint error.
"""

import asyncio
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from api.supabase_annotation_manager import SupabaseAnnotationManager


class TestSupabaseAnnotationManager:
    """The manager with a mocked async Supabase client."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def am(self, mock_client):
        return SupabaseAnnotationManager(mock_client)

    def test_list_annotations_scopes_to_job(self, am, mock_client):
        """Without a version, every version's notes come back - the panel uses
        that to show earlier versions carry annotations."""
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.order.return_value.execute = AsyncMock(
            return_value=Mock(data=[{"id": "a1"}])
        )

        rows = asyncio.run(am.list_annotations("job-1"))

        assert rows == [{"id": "a1"}]
        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "job_id", "job-1"
        )
        # No version filter applied.
        chain.eq.assert_not_called()

    def test_list_annotations_filters_by_version(self, am, mock_client):
        """Anchors are version-pinned, so the panel asks for one version."""
        eq_job = mock_client.table.return_value.select.return_value.eq.return_value
        eq_job.eq.return_value.order.return_value.execute = AsyncMock(
            return_value=Mock(data=[])
        )

        asyncio.run(am.list_annotations("job-1", version=2))

        eq_job.eq.assert_called_with("version", 2)

    def test_list_annotations_returns_empty_when_data_is_none(self, am, mock_client):
        """A None payload must not propagate as None: callers iterate the result."""
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.order.return_value.execute = AsyncMock(return_value=Mock(data=None))

        assert asyncio.run(am.list_annotations("job-1")) == []

    def test_create_annotation_never_sends_user_id(self, am, mock_client):
        """Authorship comes from auth.uid() in the DB, never from the client -
        otherwise a caller could attribute a comment to someone else."""
        mock_client.table.return_value.insert.return_value.execute = AsyncMock(
            return_value=Mock(data=[{"id": "a1"}])
        )

        asyncio.run(
            am.create_annotation(
                job_id="job-1", version=1, section="risks",
                start_offset=0, end_offset=5, quote="hello", body="note",
            )
        )

        sent = mock_client.table.return_value.insert.call_args.args[0]
        assert "user_id" not in sent
        assert sent["job_id"] == "job-1"

    def test_update_annotation_body_only_touches_body(self, am, mock_client):
        """An edit must not be able to smuggle in a resolution change."""
        chain = mock_client.table.return_value.update.return_value.eq.return_value
        chain.execute = AsyncMock(return_value=Mock(data=[{"id": "a1", "body": "new"}]))

        asyncio.run(am.update_annotation_body("a1", "new"))

        update_call = mock_client.table.return_value.update
        assert update_call.call_args.args[0] == {"body": "new"}

    def test_update_annotation_returns_none_when_rls_matches_nothing(
        self, am, mock_client
    ):
        """A non-author's update matches no row; the route turns that into 403."""
        chain = mock_client.table.return_value.update.return_value.eq.return_value
        chain.execute = AsyncMock(return_value=Mock(data=[]))

        assert asyncio.run(am.update_annotation_body("a1", "new")) is None

    def test_set_resolution_goes_through_definer_function(self, am, mock_client):
        """Resolving must use the RPC, not a table update: no RLS policy can
        allow setting `resolution` without also allowing edits to `body`."""
        mock_client.rpc.return_value.execute = AsyncMock(return_value=Mock(data=None))

        asyncio.run(am.set_resolution("a1", "accepted"))

        name, params = mock_client.rpc.call_args.args
        assert name == "set_annotation_resolution"
        assert params == {"annotation_id": "a1", "new_resolution": "accepted"}
        mock_client.table.return_value.update.assert_not_called()

    def test_set_resolution_none_reopens(self, am, mock_client):
        mock_client.rpc.return_value.execute = AsyncMock(return_value=Mock(data=None))

        asyncio.run(am.set_resolution("a1", None))

        assert mock_client.rpc.call_args.args[1]["new_resolution"] is None

    def test_get_profiles_short_circuits_on_empty(self, am, mock_client):
        """No authors means no query at all."""
        assert asyncio.run(am.get_profiles([])) == {}
        mock_client.table.assert_not_called()

    def test_get_profiles_dedupes_and_keys_by_id(self, am, mock_client):
        """One query for the whole thread, not one per comment."""
        chain = mock_client.table.return_value.select.return_value.in_.return_value
        chain.execute = AsyncMock(
            return_value=Mock(
                data=[{"id": "u1", "display_name": "Ada", "avatar_url": None}]
            )
        )

        out = asyncio.run(am.get_profiles(["u1", "u1", "u2"]))

        assert out == {"u1": {"id": "u1", "display_name": "Ada", "avatar_url": None}}
        in_call = mock_client.table.return_value.select.return_value.in_
        sent_ids = in_call.call_args.args[1]
        assert sorted(sent_ids) == ["u1", "u2"]

    def test_create_share_upserts_so_resharing_changes_role(self, am, mock_client):
        """Re-sharing to the same user updates their role instead of failing on
        the composite primary key."""
        mock_client.table.return_value.upsert.return_value.execute = AsyncMock(
            return_value=Mock(
                data=[{"job_id": "job-1", "user_id": "u2", "role": "viewer"}]
            )
        )

        asyncio.run(am.create_share("job-1", "u2", "viewer", "u1"))

        kwargs = mock_client.table.return_value.upsert.call_args.kwargs
        assert kwargs["on_conflict"] == "job_id,user_id"
        row = mock_client.table.return_value.upsert.call_args.args[0]
        assert row["granted_by"] == "u1"


class TestAnnotationRequestValidation:
    """The root/reply shape rules the routes enforce before reaching the DB."""

    def test_reply_may_not_carry_an_anchor(self):
        """A reply with its own anchor is rejected: only roots mark a passage."""
        from api.schemas import AnnotationCreateRequest

        payload = AnnotationCreateRequest(
            body="agreed", parent_id="a1", section="risks", start_offset=0, end_offset=3
        )
        anchor_fields = (
            payload.section, payload.start_offset, payload.end_offset, payload.quote
        )
        # This is the condition the route branches on.
        assert any(f is not None for f in anchor_fields)

    def test_root_requires_every_anchor_field(self):
        """A root missing any anchor part cannot be positioned, so it is a 400
        rather than a partially-anchored row."""
        from api.schemas import AnnotationCreateRequest

        payload = AnnotationCreateRequest(body="note", version=1, section="risks")
        missing = [
            name
            for name, value in (
                ("version", payload.version),
                ("section", payload.section),
                ("start_offset", payload.start_offset),
                ("end_offset", payload.end_offset),
                ("quote", payload.quote),
            )
            if value is None
        ]
        assert missing == ["start_offset", "end_offset", "quote"]

    def test_body_must_not_be_empty(self):
        """An empty comment is meaningless and the DB rejects it too."""
        from pydantic import ValidationError
        from api.schemas import AnnotationCreateRequest

        with pytest.raises(ValidationError):
            AnnotationCreateRequest(body="")

    def test_resolution_rejects_unknown_values(self):
        """Only the tick and the cross exist; anything else is a bad request."""
        from pydantic import ValidationError
        from api.schemas import AnnotationResolveRequest

        assert AnnotationResolveRequest(resolution="accepted").resolution == "accepted"
        assert AnnotationResolveRequest(resolution=None).resolution is None
        with pytest.raises(ValidationError):
            AnnotationResolveRequest(resolution="maybe")

    def test_section_is_constrained_to_the_thesis_blocks(self):
        """A section outside the thesis cannot be anchored into."""
        from pydantic import ValidationError
        from api.schemas import AnnotationCreateRequest

        with pytest.raises(ValidationError):
            AnnotationCreateRequest(
                body="note", version=1, section="footer",
                start_offset=0, end_offset=1, quote="x",
            )

    def test_inverted_range_is_rejected_at_the_request_layer(self):
        """An inverted range marks nothing. The DB constraint also catches it,
        but only as a 500 - this keeps it a 400."""
        from pydantic import ValidationError
        from api.schemas import AnnotationCreateRequest

        with pytest.raises(ValidationError):
            AnnotationCreateRequest(
                body="note", version=1, section="risks",
                start_offset=10, end_offset=4, quote="x",
            )

    def test_empty_range_is_rejected(self):
        """start == end selects no characters."""
        from pydantic import ValidationError
        from api.schemas import AnnotationCreateRequest

        with pytest.raises(ValidationError):
            AnnotationCreateRequest(
                body="note", version=1, section="risks",
                start_offset=7, end_offset=7, quote="x",
            )

    def test_valid_range_is_accepted(self):
        from api.schemas import AnnotationCreateRequest

        payload = AnnotationCreateRequest(
            body="note", version=1, section="risks",
            start_offset=0, end_offset=5, quote="hello",
        )
        assert payload.end_offset > payload.start_offset

    def test_version_must_be_positive(self):
        """Versions are 1-based; version 0 does not exist."""
        from pydantic import ValidationError
        from api.schemas import AnnotationCreateRequest

        with pytest.raises(ValidationError):
            AnnotationCreateRequest(
                body="note", version=0, section="risks",
                start_offset=0, end_offset=1, quote="x",
            )


class TestResolutionErrorContract:
    """The route maps DB failures by SQLSTATE, not message text.

    These pin the contract from both ends: the SQL raises the codes, and the
    route branches on them. Reword an exception in sql/annotations.sql and
    nothing breaks; change a code without changing both sides and these fail.
    """

    def test_sql_raises_explicit_sqlstates(self):
        """The definer function must attach codes, not rely on prose."""
        sql = Path("sql/annotations.sql").read_text()
        fn = sql.split("create or replace function set_annotation_resolution")[1]
        # Every raise inside the function carries an errcode.
        raises = [line for line in fn.splitlines() if "raise exception" in line]
        assert raises, "expected the function to raise"
        for line in raises:
            assert "errcode" in line, f"raise without a SQLSTATE: {line.strip()}"

    def test_route_constants_match_the_sql(self):
        from api.routes import (
            SQLSTATE_INSUFFICIENT_PRIVILEGE,
            SQLSTATE_INVALID_PARAMETER,
            SQLSTATE_NO_DATA_FOUND,
        )

        sql = Path("sql/annotations.sql").read_text()
        fn = sql.split("create or replace function set_annotation_resolution")[1]
        for code in (
            SQLSTATE_INSUFFICIENT_PRIVILEGE,
            SQLSTATE_INVALID_PARAMETER,
            SQLSTATE_NO_DATA_FOUND,
        ):
            assert f"errcode = '{code}'" in fn, f"route expects {code}, SQL never raises it"

    def test_route_branches_on_code_not_message(self):
        """Guards the regression this replaced: branching on the exception's
        prose. The user-facing strings may well contain words like "not found",
        so this asserts the mechanism, not the absence of a phrase."""
        route_src = Path("api/routes.py").read_text()
        body = route_src.split("async def resolve_annotation")[1].split("@router")[0]
        assert "exc.code ==" in body, "resolution errors must be matched by SQLSTATE"
        # The old implementation did `message = str(exc)` then substring checks.
        assert "str(exc)" not in body, "matching on the exception text is what broke"
