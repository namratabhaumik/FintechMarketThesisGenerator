"""Per-user auth: verify Supabase access tokens (JWKS / ES256) and build a
per-request, user-scoped Supabase client so Row Level Security applies.

The service-role client (see api.deps) bypasses RLS and is for admin/maintenance
only; user-facing job endpoints depend on get_user_job_manager instead, which
runs every query as the caller (anon key + their JWT).

Client pooling: Reuses Supabase SDK clients across requests to reduce creation
overhead. The pool maintains a bounded number of clients; excess requests queue
until one becomes available.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException
from jwt import PyJWKClient
from supabase import acreate_client

from api.supabase_job_manager import SupabaseJobManager
from core.interfaces.job_manager import IJobManager

logger = logging.getLogger(__name__)

_jwks_client: PyJWKClient | None = None
_client_pool: "SupabaseClientPool | None" = None


class SupabaseClientPool:
    """Bounded pool of reusable Supabase async clients.

    Maintains a fixed number of idle clients. When a request needs a client:
      - If idle clients exist, reuse one (fast)
      - If pool is full, wait for one to be released (backpressure)
      - If pool has space, create a new one
    This reduces the per-request handshake overhead while bounding memory.
    """

    def __init__(self, pool_size: int = 20, supabase_url: str = "", anon_key: str = ""):
        self.pool_size = pool_size
        self.supabase_url = supabase_url
        self.anon_key = anon_key
        self.semaphore = asyncio.Semaphore(pool_size)
        self._idle_clients: list = []
        logger.info(f"Initialized SupabaseClientPool with size {pool_size}")

    async def acquire(self):
        """Get a client from the pool (create if needed, or wait if exhausted)."""
        await self.semaphore.acquire()
        try:
            if self._idle_clients:
                client = self._idle_clients.pop()
                logger.debug("Reusing client from pool")
            else:
                logger.debug("Creating new client (pool at capacity)")
                client = await acreate_client(self.supabase_url, self.anon_key)
            return client
        except BaseException:
            # Creating the client failed (or the task was cancelled) after the
            # semaphore slot was taken. Release it, otherwise a transient failure
            # permanently shrinks the pool and enough failures deadlock it.
            self.semaphore.release()
            raise

    async def release(self, client):
        """Return a client to the pool for reuse."""
        self._idle_clients.append(client)
        self.semaphore.release()
        logger.debug(f"Released client back to pool (idle: {len(self._idle_clients)})")

    async def shutdown(self):
        """Close all idle clients on app shutdown."""
        for client in self._idle_clients:
            try:
                await client.postgrest.aclose()
            except Exception as e:
                logger.warning(f"Error closing pooled client on shutdown: {e}")
        self._idle_clients.clear()
        logger.info("SupabaseClientPool shutdown complete")


def init_client_pool(pool_size: int = 20):
    """Initialize the global client pool at app startup."""
    global _client_pool
    if _client_pool is not None:
        return
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    _client_pool = SupabaseClientPool(pool_size, supabase_url, anon_key)


async def shutdown_client_pool():
    """Shutdown the global client pool."""
    global _client_pool
    if _client_pool:
        await _client_pool.shutdown()
        _client_pool = None


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
    The client is acquired from the pool and released back when the request finishes."""
    if _client_pool is None:
        raise RuntimeError("Client pool not initialized. Call init_client_pool() at startup.")

    client = await _client_pool.acquire()
    # SECURITY: pooled clients are shared across users. This auth() call re-scopes
    # the client to the current caller and MUST run after every acquire(), before
    # any query. A pooled client still carries the previous caller's token until
    # this line runs, so any future pool consumer must do the same or it would run
    # queries as whoever used the client last.
    client.postgrest.auth(user.token)
    try:
        yield SupabaseJobManager(client)
    finally:
        await _client_pool.release(client)