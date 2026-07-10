"""Per-user auth: verify Supabase access tokens (JWKS / ES256) and build a
per-request, user-scoped Supabase client so Row Level Security applies.

The service-role client (see api.deps) bypasses RLS and is for admin/maintenance
only; user-facing job endpoints depend on get_user_job_manager instead, which
runs every query as the caller (anon key + their JWT).
"""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException
from jwt import PyJWKClient
from supabase import acreate_client

from api.supabase_job_manager import SupabaseJobManager
from core.interfaces.job_manager import IJobManager


_jwks_client: PyJWKClient | None = None


def _supabase_url() -> str:
    return os.getenv("SUPABASE_URL", "").rstrip("/")


def _jwks() -> PyJWKClient:
    """Lazy, cached JWKS client: fetches the project's public keys on first use
    and caches the set, selecting the right key per token by `kid`."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(f"{_supabase_url()}/auth/v1/.well-known/jwks.json")
    return _jwks_client


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(status_code=401, detail={"code": "unauthorized", "message": message})


@dataclass
class AuthUser:
    """The authenticated caller."""
    id: str      # Supabase user UUID (JWT `sub`)
    token: str   # raw access token, forwarded to PostgREST so RLS sees auth.uid()


def get_current_user(authorization: str | None = Header(default=None)) -> AuthUser:
    """Verify the Supabase access token and return the user.

    RLS re-validates the same token at the database, but verifying here fails
    fast with a clean 401 and yields the user id up front.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _unauthorized("Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        key = _jwks().get_signing_key_from_jwt(token).key
        claims = pyjwt.decode(
            token,
            key,
            algorithms=["ES256"],
            audience="authenticated",
        )
    except Exception:
        raise _unauthorized("Invalid or expired token")
    sub = claims.get("sub")
    if not sub:
        raise _unauthorized("Token has no subject")
    return AuthUser(id=sub, token=token)


async def get_user_job_manager(
    user: AuthUser = Depends(get_current_user),
) -> AsyncIterator[IJobManager]:
    """A per-request job manager backed by a user-scoped Supabase client (anon
    key + the caller's JWT), so every jobs query runs under RLS as that user.
    The client is closed when the request finishes."""
    client = await acreate_client(_supabase_url(), os.getenv("SUPABASE_ANON_KEY", ""))
    client.postgrest.auth(user.token)
    try:
        yield SupabaseJobManager(client)
    finally:
        await client.postgrest.aclose()