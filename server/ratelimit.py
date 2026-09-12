"""
Lightweight in-process rate limiting.

A sliding-window log limiter, keyed by client IP. No external dependencies and
no Redis — fine for the single-process server this project runs. If you ever
scale to multiple workers you'll want a shared store instead, since each worker
keeps its own counters.

    limiter = RateLimiter(limit=30, window=60)        # 30 requests / minute
    allowed, remaining, retry_after = limiter.hit("1.2.3.4", now=time.time())
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimiter:
    """Sliding-window request counter shared across requests (thread-safe)."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, now: float) -> Tuple[bool, int, float]:
        """Record a request for `key`. Returns (allowed, remaining, retry_after).

        `retry_after` is seconds until the oldest in-window hit expires (0 when
        the request is allowed).
        """
        cutoff = now - self.window
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= self.limit:
                retry_after = q[0] + self.window - now
                return False, 0, max(0.0, retry_after)

            q.append(now)
            return True, self.limit - len(q), 0.0


def install_rate_limit(
    app, limiter: RateLimiter, *, protect_prefix: tuple[str, ...] = ("/v1", "/api/v1")
) -> None:
    """Attach `limiter` to a FastAPI/Starlette app as HTTP middleware.

    Only paths under one of `protect_prefix` are limited (so /healthz stays
    open). Every routed prefix must be listed, or it silently goes unlimited. On
    every response we set the standard `X-RateLimit-*` headers; over-limit
    requests get a 429 with `Retry-After` and an OpenAI-shaped error body.
    """

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if not request.url.path.startswith(protect_prefix):
            return await call_next(request)

        key = _client_key(request)
        now = time.time()
        allowed, remaining, retry_after = limiter.hit(key, now)

        if not allowed:
            resp = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": (
                            f"Rate limit exceeded: {limiter.limit} requests per "
                            f"{int(limiter.window)}s. Retry in "
                            f"{retry_after:.1f}s."
                        ),
                        "type": "rate_limit_error",
                    }
                },
            )
            resp.headers["Retry-After"] = str(int(retry_after) + 1)
        else:
            resp = await call_next(request)

        resp.headers["X-RateLimit-Limit"] = str(limiter.limit)
        resp.headers["X-RateLimit-Remaining"] = str(remaining)
        resp.headers["X-RateLimit-Reset"] = str(int(now + limiter.window))
        return resp


TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "1").lower() in ("1", "true", "yes")
TRUSTED_PROXIES = {
    ip.strip() for ip in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if ip.strip()
}


def _client_key(request: Request) -> str:
    """Client identity: X-Forwarded-For / CF-Connecting-IP only if peer is trusted proxy, else peer IP."""
    peer_ip = request.client.host if request.client else "unknown"
    if TRUST_PROXY_HEADERS and (peer_ip in TRUSTED_PROXIES or peer_ip.startswith("127.") or peer_ip == "::1"):
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return peer_ip
