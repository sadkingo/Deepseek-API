"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import debuglog
from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient, ServerBusy

from .config import (
    CORS_ORIGINS,
    resolve_alias,
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    model_thinking,
    resolve_model_type,
)
from .openai_format import (
    completion_response,
    extract_tool_call,
    message_images,
    message_texts,
    messages_to_prompt,
    serialize_tool_call,
    stream_chunks,
    strip_role_leak,
    tools_preamble,
)
from .threads import ThreadCache, history_key
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest

load_dotenv()

# `kill -USR1 <pid>` appends every thread's stack to logs/stacks.log. This is
# the way to see where a wedged server is blocked: ptrace is restricted on this
# machine (yama ptrace_scope=1), so py-spy needs root, while this needs nothing.
try:
    _stacks_path = debuglog.LOG_FILE.parent / "stacks.log"
    _stacks_path.parent.mkdir(parents=True, exist_ok=True)
    _stacks_file = open(_stacks_path, "a")
    faulthandler.enable(file=_stacks_file)  # also dump on hard crashes
    faulthandler.register(signal.SIGUSR1, file=_stacks_file, all_threads=True)
except (OSError, ValueError, AttributeError):
    pass  # non-main thread, read-only fs, or no SIGUSR1 (Windows) — skip

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# Browser clients need CORS headers (and a preflight handler) to call this from
# a web page. Off unless CORS_ORIGINS is set, so the default stays closed.
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()

# Lets a resent OpenAI history resume its DeepSeek thread instead of being
# replayed as a transcript. See server/threads.py.
_threads = ThreadCache()


# If every upstream slot sits occupied with zero progress for this long, the
# shared httpx client is declared wedged and rebuilt (see _watchdog below).
UPSTREAM_WEDGE_TIMEOUT = float(os.getenv("UPSTREAM_WEDGE_TIMEOUT", "120"))


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _watchdog() -> None:
    """Self-heal from a wedged upstream client.

    We once hit a deadlock inside httpcore's sync connection pool (its lock was
    orphaned while tearing down an abandoned streaming response), which froze
    every /chat/completions forever while /healthz stayed green. A poisoned lock
    can't be repaired, so when the client looks wedged we drop it and let the
    next request build a fresh one. The stuck threads are abandoned with it —
    a bounded leak (at most MAX_CONCURRENCY threads per event), vastly better
    than a dead server. Do NOT call .close() on the old client here: that would
    try to take the same poisoned lock and wedge the watchdog too.
    """
    global _client
    while True:
        time.sleep(10)
        client = _client
        if client is None or not client.wedged(UPSTREAM_WEDGE_TIMEOUT):
            continue
        debuglog.log_error(
            f"upstream client wedged (no progress for {UPSTREAM_WEDGE_TIMEOUT:.0f}s "
            "with all slots busy) — discarding it and rebuilding on next request"
        )
        with _client_lock:
            if _client is client:
                _client = None


threading.Thread(target=_watchdog, name="upstream-watchdog", daemon=True).start()


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    debuglog.log_error(f"{err_type}: {message}")
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


@app.middleware("http")
async def _log_completions(request, call_next):
    if not debuglog.ENABLED or not request.url.path.endswith("/chat/completions"):
        return await call_next(request)
    # Starlette caches the body, so reading it here does not consume it.
    body = await request.body()
    debuglog.log_request(request.method, request.url.path, body, request.headers)
    response = await call_next(request)
    if response.status_code >= 400:
        debuglog.log_error(f"HTTP {response.status_code}")
    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
@app.get("/api/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    req.model = resolve_alias(req.model)

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    history = message_texts(req.messages)

    # Prefer continuing the DeepSeek thread this history already belongs to, so
    # only the newest turn is sent and the model never sees a transcript to
    # carry on. Falls back to flattening when the thread is unknown to us.
    conversation_id = req.conversation_id
    if conversation_id is None and len(history) > 1:
        conversation_id = _threads.get(history_key(history[:-1]))

    if conversation_id:
        # Resuming: send only the new turn. That is the last message with text,
        # skipping any trailing empty or non-text (e.g. image-only) entries.
        prompt = next((t for _, t in reversed(history) if t.strip()), "")
        # Images from earlier turns were already attached when those turns were
        # sent, so on resume only the newest message's images are uploaded.
        images = message_images(req.messages[-1:])
    else:
        prompt = messages_to_prompt(req.messages)
        images = message_images(req.messages)
        # New thread with tools: teach the model the emulated tool-call
        # protocol. It goes LAST, after the client's own (often long) system
        # prompt — buried at the top it loses out to whatever tool syntax that
        # prompt implies, and the model falls back to ReAct. On resume the
        # thread already carries it.
        if req.tools:
            prompt = f"{prompt}\n\n{tools_preamble(req.tools)}"

    if not prompt.strip() and not images:
        return _error(
            "No text or image content in `messages` to send.",
            status=400, err_type="invalid_request_error",
        )

    debuglog.log_prompt(prompt, conversation_id, resumed=bool(conversation_id))

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if conversation_id else resolve_model_type(req.model)

    # DeepThink: on when the request asks for it OR the model id bakes it in
    # ("-reasoner" ids exist for frontends that can only vary the model name).
    # Unlike the model, thinking is per-message, so it applies on resume too.
    thinking = req.thinking or model_thinking(req.model)

    def remember(reply_text: str, cid: str | None, streamed: bool = False) -> None:
        """Record the thread so the client's next resend resumes it."""
        debuglog.log_reply(reply_text, cid, streamed=streamed)
        if cid:
            _threads.put(history_key(list(history) + [("assistant", reply_text)]), cid)

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    def upload_images() -> list:
        """Upload the request's images to DeepSeek, returning their file ids.

        Blocking (upload + parse-poll per image), so it must run off the event
        loop — inside the stream generator or via run_in_threadpool.
        """
        return [client.upload_file(data, filename, mime)
                for filename, mime, data in images]

    if req.stream:
        def gen():
            try:
                stream = client.stream(
                    prompt, conversation_id=conversation_id,
                    model=model_type, thinking=thinking, search=req.search,
                    ref_file_ids=upload_images(),
                )
                yield from stream_chunks(
                    req.model, stream,
                    on_done=lambda t, c: remember(t, c, streamed=True),
                    tools_enabled=bool(req.tools),
                )
            except Exception as e:
                # Headers are already sent, so the failure has to travel as an
                # SSE frame; otherwise the client just sees a truncated stream.
                debuglog.log_error(f"stream failed: {e!r}")
                payload = {"error": {"message": f"DeepSeek request failed: {e}",
                                     "type": "server_error"}}
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    def run_chat():
        return client.chat(prompt, conversation_id, model_type,
                           thinking, req.search, upload_images())

    try:
        reply = await run_in_threadpool(run_chat)
    except ServerBusy as e:
        return _error(str(e), status=503, err_type="overloaded_error")
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    # Split off an emulated tool call before leak-stripping: tool arguments may
    # embed file contents whose lines would otherwise look like a leaked turn.
    tool_call = None
    text = reply.text
    reasoning = reply.thinking
    if req.tools:
        text, tool_call = extract_tool_call(text)
        if reasoning and not tool_call and not text.strip():
            # With DeepThink on, the model sometimes ends its reasoning with the
            # call and writes no reply. Only rescue it when the reply really is
            # empty, so a call merely mentioned mid-thought is never run — and
            # in that case leave `reasoning` untouched so none of it is lost.
            reasoning_text, think_call = extract_tool_call(reasoning, strict=True)
            if think_call:
                tool_call, reasoning = think_call, reasoning_text
    text = strip_role_leak(text)

    if not text.strip() and not tool_call:
        # Nothing actionable came back. Either DeepSeek said nothing at all
        # (how it answers while throttling the account — the reply returns
        # almost instantly), or it reasoned and then stopped without writing a
        # reply. Both leave the caller with a turn it cannot act on, so report
        # a retryable error rather than a blank message that stalls an agent.
        return _error(
            "DeepSeek returned no reply"
            f"{' (it produced only reasoning)' if reasoning else ''}. This is "
            "usually transient, or the account being throttled for sending too "
            "many requests; wait a moment and retry.",
            status=503, err_type="overloaded_error",
        )

    if tool_call:
        # Record the reply in its protocol text form, matching what the client
        # resends as an assistant message carrying tool_calls.
        call_text = serialize_tool_call(*tool_call)
        remembered = f"{text}\n{call_text}".strip() if text else call_text
    else:
        remembered = text
    remember(remembered, reply.conversation_id)
    return completion_response(req.model, text, prompt, reply.conversation_id,
                               reasoning=reasoning, tool_call=tool_call)
