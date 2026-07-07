"""Security primitives for the API: an optional shared-key gate and a per-IP
rate limiter.

Env knobs (all optional):
    FINTHESIS_API_KEY         Shared key for mutating endpoints. Unset -> gate
                              off (local dev).
    RATE_LIMIT_STORAGE_URI    limits backend. Default "memory://" (per-worker,
                              fine for one instance). Set "redis://host:6379/0"
                              to share buckets across instances.
    RATE_LIMIT_DEFAULT        Optional global ceiling for every route (e.g.
                              "120/hour"). Unset -> only the per-route limits
                              below apply. Enabling it also enables
                              SlowAPIMiddleware in main.py; /health is exempt.
    RATE_LIMIT_GENERATE       Per-IP limit on thesis generation. Default "10/minute".
    RATE_LIMIT_REFINE         Per-IP limit on refinement.        Default "20/minute".
"""

import os
import secrets

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

API_KEY_HEADER = "X-API-Key"
API_KEY_ENV = "FINTHESIS_API_KEY"


# --- Rate limiter -------------------------------------------------------------

def _rate_limit_key(request: Request) -> str:
    """per-user limiting on a shared bucket for all endpoints (frontend surfaces 
    a single "rate limit exceeded" error). This is the default key_func for the 
    Limiter instance below."""
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


# --- Shared-key gate ----------------------------------------------------------

def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """Gate mutating endpoints behind a shared key when one is configured.

    No-op when FINTHESIS_API_KEY is unset. Raises 401 when the header is missing or invalid.
    """
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "Missing or invalid API key"},
        )