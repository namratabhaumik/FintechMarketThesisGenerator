"""Security primitives for the API: a per-IP rate limiter.

(Per-user auth is handled separately in api.auth via Supabase JWTs, which
replaced the earlier shared-key cost gate.)

Env knobs (all optional):
    RATE_LIMIT_STORAGE_URI    limits backend. Default "memory://" (per-worker,
                              fine for one instance). Set "redis://host:6379/0"
                              to share buckets across instances.
    RATE_LIMIT_DEFAULT        Optional global ceiling for every route (e.g.
                              "120/hour"). Unset -> only the per-route limits
                              below apply. Enabling it also enables
                              SlowAPIMiddleware in main.py; /health is exempt.
    RATE_LIMIT_TRUSTED_HOPS   How many reverse proxies sit in front of the app.
                              Default 1 (Render). Decides how far from the
                              right-hand end of X-Forwarded-For the client IP is
                              read; see _rate_limit_key.
    RATE_LIMIT_GENERATE       Per-IP limit on thesis generation. Default "10/minute".
    RATE_LIMIT_REFINE         Per-IP limit on refinement.        Default "20/minute".
    RATE_LIMIT_ANNOTATE       Per-IP limit on annotation writes. Default "60/minute".
"""

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


# --- Rate limiter -------------------------------------------------------------

# Proxies we sit behind. Render alone is 1. See _rate_limit_key for why this
# counts from the right-hand end of X-Forwarded-For.
TRUSTED_PROXY_HOPS = max(1, int(os.getenv("RATE_LIMIT_TRUSTED_HOPS", "1")))


def _rate_limit_key(request: Request) -> str:
    """per-user limiting on a shared bucket for all endpoints (frontend surfaces
    a single "rate limit exceeded" error). This is the default key_func for the
    Limiter instance below.

    Behind a reverse proxy, the TCP peer is the proxy, so keying
    on it would put every user in ONE bucket and let a single client's burst
    429 everyone. Prefer the client IP the proxy records in X-Forwarded-For;
    absent that header (local dev, direct access), fall back to the peer
    address.

    Reads from the RIGHT, not the left. X-Forwarded-For is client-supplied
    until a proxy overwrites it, and Render's own docs contradict themselves on
    whether it overwrites or merely appends -- so the leftmost entry may be a
    value the caller invented to get a fresh rate-limit bucket per request.
    Entries are appended left-to-right, so the LAST one was written by the hop
    closest to us (the only one we can vouch for). The rightmost read is correct
    under either Render behaviour: if it appends, the last entry is the real
    peer; if it overwrites, there is one entry and last == first.

    RATE_LIMIT_TRUSTED_HOPS counts the proxies we run behind (1 for Render
    alone; raise it if a CDN is added in front, or the CDN's appended entry
    becomes the key and buckets collapse to one per edge node). A chain shorter
    than that means the request did not arrive through the expected proxies, so
    trust none of it and key on the peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if len(hops) >= TRUSTED_PROXY_HOPS:
            return hops[-TRUSTED_PROXY_HOPS]
    return get_remote_address(request)


_default_limit = os.getenv("RATE_LIMIT_DEFAULT")
_default_limits = [_default_limit] if _default_limit else []

# When True, main.py adds SlowAPIMiddleware so _default_limits apply globally.
GLOBAL_RATE_LIMIT_ENABLED = bool(_default_limits)

limiter = Limiter(
    key_func=_rate_limit_key,
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
    default_limits=_default_limits,
)

# Per-route limits for the cost-bearing (LLM) endpoints, tunable without a deploy.
GENERATE_LIMIT = os.getenv("RATE_LIMIT_GENERATE", "10/minute")
REFINE_LIMIT = os.getenv("RATE_LIMIT_REFINE", "20/minute")

# Reads are unlimited: the panel loads on every thesis open.
ANNOTATE_LIMIT = os.getenv("RATE_LIMIT_ANNOTATE", "60/minute")


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Render a 429 in the same {detail: {code, message}} shape as other errors,
    so the frontend's ApiError parsing surfaces it like any other API error."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": {
                "code": "rate_limited",
                "message": f"Rate limit exceeded: {exc.detail}",
            }
        },
    )