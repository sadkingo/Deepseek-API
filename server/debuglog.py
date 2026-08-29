"""Request/response logging for debugging what a client actually sends.

Writes to stdout and to a rotating file (default `logs/requests.log`). Enabled
by default; set LOG_REQUESTS=0 to turn it off.

The log records prompt text, so treat the file as sensitive: it holds whatever
users type. LOG_MAX_BODY caps how much of each body is written.
"""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

ENABLED = os.getenv("LOG_REQUESTS", "1").lower() not in ("0", "false", "no", "off")
LOG_FILE = Path(os.getenv("LOG_FILE", ROOT / "logs" / "requests.log"))
MAX_BODY = int(os.getenv("LOG_MAX_BODY", "4000"))

_log: Optional[logging.Logger] = None


def logger() -> logging.Logger:
    global _log
    if _log is not None:
        return _log

    lg = logging.getLogger("deepseek.requests")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    lg.addHandler(stream)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
        rotating.setFormatter(fmt)
        lg.addHandler(rotating)
    except OSError as e:  # read-only dir, bad path - stdout logging still works
        lg.warning("[log] file logging disabled: %s", e)

    _log = lg
    return lg


def _clip(text: str, limit: int = MAX_BODY) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... (+{len(text) - limit} chars)"


def log_request(method: str, path: str, body: bytes, headers) -> None:
    """Log an incoming request exactly as it arrived."""
    if not ENABLED:
        return
    lg = logger()
    auth = headers.get("authorization")
    lg.info(
        "\n--> %s %s  origin=%s  auth=%s  content-type=%s",
        method, path,
        headers.get("origin") or "-",
        # Never log the token itself.
        f"present ({len(auth)} chars)" if auth else "none",
        headers.get("content-type") or "-",
    )

    raw = body.decode("utf-8", "replace")
    lg.info("    raw body: %s", _clip(raw))

    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        lg.info("    (body is not valid JSON)")
        return

    if not isinstance(obj, dict):
        return

    msgs = obj.get("messages")
    lg.info(
        "    model=%r stream=%r conversation_id=%r messages=%s",
        obj.get("model"), obj.get("stream"), obj.get("conversation_id"),
        len(msgs) if isinstance(msgs, list) else "MISSING",
    )
    if isinstance(msgs, list):
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                lg.info("      [%d] not an object: %r", i, m)
                continue
            content = m.get("content")
            if isinstance(content, str):
                shape = f"str({len(content)})"
                preview = _clip(content, 200)
            elif content is None:
                shape, preview = "null", ""
            elif isinstance(content, list):
                types = [p.get("type") if isinstance(p, dict) else type(p).__name__
                         for p in content]
                shape, preview = f"parts{types}", _clip(json.dumps(content)[:200], 200)
            else:
                shape, preview = type(content).__name__, repr(content)[:200]
            lg.info("      [%d] role=%-9s content=%-12s %s",
                    i, m.get("role"), shape, preview)


def log_prompt(prompt: str, conversation_id: Optional[str], resumed: bool) -> None:
    """Log what will actually be sent upstream to DeepSeek."""
    if not ENABLED:
        return
    logger().info(
        "    => %s  cid=%s  prompt(%d): %s",
        "RESUME thread" if resumed else "NEW thread",
        conversation_id or "-", len(prompt), _clip(prompt, 600),
    )


def log_reply(text: str, conversation_id: Optional[str], streamed: bool = False) -> None:
    if not ENABLED:
        return
    logger().info(
        "<-- reply%s len=%d cid=%s: %s",
        " (streamed)" if streamed else "", len(text), conversation_id or "-",
        _clip(text, 400),
    )


def log_error(message: str) -> None:
    if not ENABLED:
        return
    logger().info("<-- ERROR %s", message)
