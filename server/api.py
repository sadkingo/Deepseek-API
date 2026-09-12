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
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import debuglog
from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient, RateLimited, ServerBusy

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
    assistant_fingerprint,
    completion_response,
    declared_tool_names,
    extract_tool_calls,
    flatten_directive,
    message_images,
    message_texts,
    messages_to_prompt,
    serialize_tool_call,
    stream_chunks,
    strip_role_leak,
    tools_fingerprint,
    tools_preamble,
    tools_reminder,
    user_labels,
)
from .threads import ThreadCache, TurnIndex
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
# replayed as a transcript. See server/threads.py. Saved next to the session
# so a restart does not turn every open conversation into a fresh thread
# (THREADS_FILE overrides the location).
_threads = TurnIndex(path=os.getenv(
    "THREADS_FILE",
    str(Path(__file__).resolve().parent.parent / "session" / "threads.json")))

# Which tool set each DeepSeek thread has been taught, keyed by chat session.
# The protocol preamble is expensive to repeat and confusing to repeat wrongly,
# so it is sent once per thread — but it MUST be resent when the caller's tools
# change, or the model keeps calling tools that are gone and never learns the
# new ones. Zed varies its tool set per request (profiles, MCP servers), so
# this is a live case, not a hypothetical.
_thread_tools = ThreadCache()


# Sent in place of a message when the client resends our own last reply and
# nothing after it: it wants that reply continued.
CONTINUE_PROMPT = ("Continue your previous reply from exactly where it stopped. "
                   "Do not repeat anything already written and do not add a "
                   "preamble; carry straight on.")


def _session_of(conversation_id: str | None) -> str:
    """The chat-session part of a conversation_id — stable across turns."""
    return (conversation_id or "").partition(":")[0]


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


class _Replayed:
    """A consumed-then-resumed stream: yields the peeked event, then the rest.

    Resuming a dead thread only fails once the request is actually made, which
    happens on the first `events()` step. Peeking one event surfaces that while
    nothing has been sent to the client yet, so the retry stays invisible.
    """

    def __init__(self, stream, peeked, rest):
        self._stream, self._peeked, self._rest = stream, peeked, rest

    def events(self):
        if self._peeked is not None:
            yield self._peeked
        yield from self._rest

    @property
    def conversation_id(self):
        return self._stream.conversation_id


# A muted account cannot be un-muted from here — the only fix is a fresh one.
# When an upstream error says "user is muted", the server drops a flag file
# and shuts itself down; bin/start (a supervisor loop) sees the flag, runs the
# sign-up automation to register a fresh account, and starts the server again
# on the new session. Fired at most once per server lifetime, and bin/start
# additionally refuses to recover twice within 10 minutes, so a muted-again
# replacement account can't spiral into endless registrations.
_MUTED_FLAG = Path(__file__).resolve().parents[1] / "logs" / ".account-muted"
_recover_launched = False
_recover_lock = threading.Lock()


def _maybe_recover_muted(message: str) -> None:
    global _recover_launched
    if "user is muted" not in message.lower():
        return
    with _recover_lock:
        if _recover_launched:
            return
        _recover_launched = True
    debuglog.log_error(
        "account muted — shutting down so bin/start can register a fresh "
        "account and restart the server"
    )
    try:
        _MUTED_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _MUTED_FLAG.touch()
    except OSError as e:
        debuglog.log_error(f"could not write {_MUTED_FLAG}: {e}")
        with _recover_lock:
            _recover_launched = False
        return

    def _shutdown():
        # Give the in-flight error response a moment to reach the client,
        # then ask uvicorn for a graceful exit.
        time.sleep(2)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_shutdown, name="muted-shutdown",
                     daemon=True).start()


def _error(message: str, status: int = 500, err_type: str = "server_error",
           retry_after: int = None):
    debuglog.log_error(f"{err_type}: {message}")
    _maybe_recover_muted(message)
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
        headers=headers,
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
    # only the new turns are sent and the model never sees a transcript to
    # carry on. Falls back to flattening when the thread is unknown to us.
    #
    # The lookup walks BACKWARDS through the history rather than testing only
    # the newest prefix. One unrecognised turn — a reply the client reworded,
    # a turn that failed and was retried — would otherwise throw away the whole
    # thread and replay the conversation as a single enormous prompt (94k
    # characters in the case this was written for), which is both wasteful and
    # the reason the model starts narrating instead of using its tools. Finding
    # a slightly older prefix costs one extra message instead.
    conversation_id = req.conversation_id
    resume_from = len(history) - 1  # messages from here on are new to the thread
    match = None
    if conversation_id is None:
        match = _threads.find(history)
        if match:
            conversation_id, resume_from = match.cid, match.resume_from
            debuglog.log_thread(
                f"matched {match.cid} ({match.describe()}); "
                f"resending {len(history) - resume_from} of {len(history)} messages")
        else:
            debuglog.log_thread(
                f"no thread matches this history of {len(history)} messages")
    # What this turn is sent from, so its result can be filed under it.
    parent_cid = conversation_id

    # Only ever announce tools the caller actually declared for this request.
    tools_fp = tools_fingerprint(req.tools)

    # The name the client prefixes the user's turns with ("sadking: ..."), so
    # a reply that goes on to write the user's next line is cut there.
    leak_labels = user_labels(req.messages)

    # The whole history as one prompt, used when starting a thread — and kept
    # around as the fallback for when a resumed thread turns out to be gone.
    fresh_prompt = messages_to_prompt(req.messages)
    if req.tools:
        fresh_prompt = f"{fresh_prompt}\n\n{tools_preamble(req.tools)}"
    # Replaying a whole agentic conversation gives the model a transcript with
    # no turn structure, which it copies instead of continuing. Say plainly
    # what is still to be done.
    directive = flatten_directive(req.messages)
    if directive:
        fresh_prompt = f"{fresh_prompt}\n\n{directive}"

    if conversation_id:
        # Resuming: send only what this thread has not seen. Usually that is
        # the newest message; after a gap it is the few messages since the last
        # turn we recognised.
        new_msgs = req.messages[resume_from:] if req.conversation_id is None \
            else req.messages[-1:]
        prompt = messages_to_prompt(new_msgs)
        if match and match.continues:
            # The thread has seen everything; the client wants the reply this
            # state produced to go on. Echoing that reply back as a prompt
            # would make the model answer its own words instead.
            prompt = CONTINUE_PROMPT
        if not prompt.strip():
            # Nothing but empty or image-only entries: fall back to the last
            # message that does carry text.
            prompt = next((t for _, t, _ in reversed(history) if t.strip()), "")
        # Images from earlier turns were already attached when those turns were
        # sent, so only the new messages' images are uploaded.
        images = message_images(new_msgs)
        # Re-teach the protocol when this thread has not been told about THIS
        # tool set: tools switched on mid-conversation, or a changed list. When
        # the set is unchanged the thread already knows it, so nothing is added.
        if req.tools:
            if _thread_tools.get(_session_of(conversation_id)) != tools_fp:
                # Not yet taught THIS tool set: switched on mid-conversation,
                # or the list changed. Send the whole thing.
                prompt = f"{prompt}\n\n{tools_preamble(req.tools, changed=True)}"
            else:
                # Already knows them, but a long session drifts away from the
                # protocol and starts describing edits instead of making them,
                # so restate the format and the obligation each turn. The
                # schemas are not repeated — the thread still has those.
                prompt = f"{prompt}\n\n{tools_reminder(req.tools)}"
    else:
        # New thread: the protocol block goes LAST, after the client's own
        # (often long) system prompt — buried at the top it loses out to
        # whatever tool syntax that prompt implies, and the model falls back
        # to ReAct.
        prompt = fresh_prompt
        images = message_images(req.messages)

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

    def remember(reply_text: str, cid: str | None, streamed: bool = False,
                 calls: list = None) -> None:
        """Record the thread so the client's next resend resumes it.

        The stored key must describe the assistant turn the way the CLIENT will
        send it back, which is why only the tool calls go in — see
        `assistant_fingerprint`.
        """
        debuglog.log_reply(reply_text, cid, streamed=streamed)
        if cid:
            fingerprint = "\n".join(serialize_tool_call(n, a)
                                     for n, a in (calls or []))
            # A reply in another session means the resume failed and the turn
            # went out as a fresh thread: file it as a root, not a child.
            parent = parent_cid if _session_of(parent_cid) == _session_of(cid) else None
            _threads.remember(history, cid, parent, fingerprint, reply_text)
            if tools_fp:
                # This thread has now seen these tools; later turns need not
                # repeat the preamble unless the set changes again.
                _thread_tools.put(_session_of(cid), tools_fp)

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

    # DeepSeek prunes chat sessions, and a cached id can also outlive a turn
    # that never really landed. Resuming one then fails outright with
    # "invalid message id" — recoverable, since the client resent the whole
    # history and we can simply start the thread over.
    def _is_stale_thread(exc: Exception) -> bool:
        text = str(exc).lower()
        return bool(conversation_id) and (
            "invalid message id" in text
            or "invalid session" in text
            or "session not found" in text
            or "chat session" in text and "not" in text
        )

    if req.stream:
        def gen():
            try:
                files = upload_images()
                stream = client.stream(
                    prompt, conversation_id=conversation_id,
                    model=model_type, thinking=thinking, search=req.search,
                    ref_file_ids=files,
                )
                try:
                    first = iter(stream.events())
                    peeked = next(first, None)
                except Exception as e:
                    if not _is_stale_thread(e):
                        raise
                    debuglog.log_error(
                        f"thread {conversation_id} is gone ({e}); "
                        "starting a new one")
                    _threads.forget(conversation_id)
                    stream = client.stream(
                        fresh_prompt, conversation_id=None,
                        model=resolve_model_type(req.model), thinking=thinking,
                        search=req.search, ref_file_ids=files,
                    )
                    peeked = None
                    first = None
                if first is not None:
                    stream = _Replayed(stream, peeked, first)
                yield from stream_chunks(
                    req.model, stream,
                    on_done=lambda t, c, cs: remember(t, c, streamed=True,
                                                     calls=cs),
                    tools_enabled=bool(req.tools),
                    known_names=declared_tool_names(req.tools),
                    leak_labels=leak_labels,
                    on_turn=lambda cs, t: debuglog.log_turn(
                        cs, t, len(req.tools or []),
                        declared_tool_names(req.tools)),
                )
            except Exception as e:
                # Headers are already sent, so the failure has to travel as an
                # SSE frame; otherwise the client just sees a truncated stream.
                debuglog.log_error(f"stream failed: {e!r}")
                # Keep the error TYPE accurate even in-stream: a client that
                # reads it can tell "back off" from "something broke".
                if isinstance(e, RateLimited):
                    err = {"message": str(e), "type": "rate_limit_error"}
                elif isinstance(e, ServerBusy):
                    err = {"message": str(e), "type": "overloaded_error"}
                else:
                    err = {"message": f"DeepSeek request failed: {e}",
                           "type": "server_error"}
                    _maybe_recover_muted(err["message"])
                yield f"data: {json.dumps({'error': err})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    def run_chat():
        return client.chat(prompt, conversation_id, model_type,
                           thinking, req.search, upload_images())

    def run_chat_fresh():
        """Same turn, but as a brand-new thread from the full history."""
        return client.chat(fresh_prompt, None, resolve_model_type(req.model),
                           thinking, req.search, upload_images())

    try:
        try:
            reply = await run_in_threadpool(run_chat)
        except Exception as e:
            if not _is_stale_thread(e):
                raise
            debuglog.log_error(
                f"thread {conversation_id} is gone ({e}); starting a new one")
            _threads.forget(conversation_id)
            reply = await run_in_threadpool(run_chat_fresh)
    except RateLimited as e:
        # DeepSeek's own limit, not ours. 429 + Retry-After is what OpenAI
        # clients (and Zed) understand as "back off and try again".
        return _error(str(e), status=429, err_type="rate_limit_error",
                      retry_after=max(30, client.pace_hint()))
    except ServerBusy as e:
        return _error(str(e), status=503, err_type="overloaded_error")
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    # Split off an emulated tool call before leak-stripping: tool arguments may
    # embed file contents whose lines would otherwise look like a leaked turn.
    tool_calls: list = []
    text = reply.text
    reasoning = reply.thinking
    if req.tools:
        names = declared_tool_names(req.tools)
        text, tool_calls = extract_tool_calls(text, known_names=names)
        if reasoning and not tool_calls:
            # With DeepThink on, the model often plans the call while thinking
            # and then writes prose about it instead of calling it. Rescue that
            # whenever the reply itself made no call — a reply that makes its
            # own call still wins.
            reasoning_text, think_calls = extract_tool_calls(
                reasoning, strict=True, known_names=names)
            if think_calls:
                tool_calls, reasoning = think_calls, reasoning_text
    text = strip_role_leak(text, leak_labels)

    if not text.strip() and not tool_calls:
        # Nothing actionable came back. Either DeepSeek said nothing at all
        # (how it answers while throttling the account — the reply returns
        # almost instantly), or it reasoned and then stopped without writing a
        # reply. Both leave the caller with a turn it cannot act on, so report
        # a retryable error rather than a blank message that stalls an agent.
        wait = client.pace_hint()
        return _error(
            "DeepSeek returned no reply"
            f"{' (it produced only reasoning)' if reasoning else ''}. This is "
            "usually transient, or the account being throttled for sending too "
            f"many requests; retry in about {wait}s.",
            status=503, err_type="overloaded_error", retry_after=wait,
        )

    if tool_calls:
        # Record the reply in its protocol text form, matching what the client
        # resends as an assistant message carrying tool_calls.
        call_text = "\n".join(serialize_tool_call(*c) for c in tool_calls)
        remembered = f"{text}\n{call_text}".strip() if text else call_text
    else:
        remembered = text
    debuglog.log_turn(tool_calls, text, len(req.tools or []),
                      declared_tool_names(req.tools))
    remember(remembered, reply.conversation_id, calls=tool_calls)
    return completion_response(req.model, text, prompt, reply.conversation_id,
                               reasoning=reasoning, tool_calls=tool_calls)
