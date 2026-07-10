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
    RATE_LIMIT_GENERATE       Per-IP limit on thesis generation. Default "10/minute".
    RATE_LIMIT_REFINE         Per-IP limit on refinement.        Default "20/minute".
"""

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


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