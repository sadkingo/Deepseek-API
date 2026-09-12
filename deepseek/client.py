"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text

    from deepseek.auth import get_session
    from deepseek.client import DeepSeekClient

    client = DeepSeekClient(get_session())
    print(client.chat("Hello!"))                 # full reply
    for chunk in client.stream("Tell a joke"):   # streamed
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import Session, get_session
from .pow import DeepSeekPow

# Everything this module learns from upstream is reported here. The server
# attaches a handler (see server/debuglog.py) so it lands in the request log;
# on its own the library stays silent, as a library should.
_log = logging.getLogger("deepseek.upstream")

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"
UPLOAD_PATH = "/api/v0/file/upload_file"
FETCH_FILES_PATH = "/api/v0/file/fetch_files"

# How long to wait for DeepSeek to parse an uploaded file (OCR etc.) before
# giving up. Small images take ~3-5 s.
FILE_PARSE_TIMEOUT = float(os.getenv("DEEPSEEK_FILE_PARSE_TIMEOUT", "60"))

# Pause before the single retry of a request DeepSeek answered with an empty
# stream (its way of saying "you are going too fast"). See _Stream._run.
EMPTY_RETRY_DELAY = float(os.getenv("DEEPSEEK_EMPTY_RETRY_DELAY", "2"))

# At most this many requests may talk to DeepSeek at once; the rest wait up to
# DEEPSEEK_QUEUE_TIMEOUT seconds for a slot and then fail fast with ServerBusy.
# One web account can't usefully serve more anyway (DeepSeek muted this account
# once already for request spam), and the bound means a burst of retries piles
# up as quick, visible errors instead of an ever-growing queue of stuck threads.
MAX_CONCURRENCY = int(os.getenv("DEEPSEEK_MAX_CONCURRENCY", "4"))
QUEUE_TIMEOUT = float(os.getenv("DEEPSEEK_QUEUE_TIMEOUT", "45"))


# Adaptive spacing between upstream requests. Nothing is spent while requests
# succeed; each throttled reply widens the gap and each success narrows it.
PACE_FIRST_DELAY = float(os.getenv("DEEPSEEK_PACE_FIRST_DELAY", "3"))
PACE_MAX_DELAY = float(os.getenv("DEEPSEEK_PACE_MAX_DELAY", "30"))


class _Pacer:
    """Keeps a minimum gap between upstream requests, sized by recent throttling.

    DeepSeek answers a throttled request by accepting it and returning an empty
    stream, so the only way to know the account is being limited is to be
    refused. A fixed delay would tax every request to avoid a problem that
    usually is not there; this stays at zero until the account actually pushes
    back, widens the gap while it keeps pushing back, and decays once replies
    come through again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gap = 0.0        # current minimum spacing, seconds
        self._next_at = 0.0    # earliest time the next request may start

    @property
    def gap(self) -> float:
        return self._gap

    def reserve(self) -> float:
        """Claim the next slot; returns how long the caller should wait."""
        with self._lock:
            now = time.time()
            start = max(now, self._next_at)
            self._next_at = start + self._gap
            return max(0.0, start - now)

    def wait(self) -> None:
        delay = self.reserve()
        if delay > 0:
            _log.info("pacing: waiting %.1fs before the next request "
                      "(gap is %.1fs after recent throttling)", delay, self._gap)
            time.sleep(delay)

    def on_success(self) -> None:
        with self._lock:
            if self._gap:
                # Halve it, and drop to zero once it is small: a recovered
                # account should stop paying for an old burst quickly.
                self._gap = 0.0 if self._gap <= PACE_FIRST_DELAY else self._gap / 2

    def on_throttled(self) -> None:
        with self._lock:
            self._gap = min(PACE_MAX_DELAY,
                            PACE_FIRST_DELAY if not self._gap else self._gap * 2)
            self._next_at = max(self._next_at, time.time() + self._gap)
            _log.warning("upstream throttled us; spacing requests %.1fs apart",
                         self._gap)


class ServerBusy(RuntimeError):
    """All upstream slots stayed occupied for the whole queue timeout."""


class RateLimited(RuntimeError):
    """DeepSeek refused the request because the account is over its limit."""

# DeepSeek's mode pill, sent as `model_type` in the completion body. "default" is
# Instant (the fast model); "expert" is the stronger, slower model. Omitting the
# field lets the backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation.

    `thinking` holds the DeepThink reasoning text when the request had
    `thinking=True` (None otherwise); it is never part of `text`.
    """

    text: str
    conversation_id: str
    thinking: Optional[str] = None

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz_error(line: str) -> Optional[str]:
    """The error message in a bare JSON body, if `line` is one.

    A rejected completion request comes back as HTTP 200 with a plain JSON
    envelope instead of an SSE stream (e.g. an empty prompt yields biz_code 6,
    "missing prompt or ref file"). Without this the stream would just look
    empty.
    """
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    data = obj.get("data") or {}
    if obj.get("code") not in (None, 0) or data.get("biz_code") not in (None, 0):
        return data.get("biz_msg") or obj.get("msg") or str(obj)
    return None


# DeepSeek reports application-level failures inside an otherwise-OK envelope:
# HTTP 200, `code` 0, and the real verdict in `data.biz_code` / `data.biz_msg`.
# 7 is "rate limit reached", which callers must be able to tell apart from a
# genuine protocol surprise so they can back off instead of retrying blindly.
_BIZ_RATE_LIMIT = 7


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")

    payload = data.get("data") or {}
    biz_code = payload.get("biz_code")
    biz_msg = payload.get("biz_msg") or ""
    if biz_code not in (None, 0):
        _log.warning("upstream rejected request: biz_code=%s biz_msg=%r",
                     biz_code, biz_msg)
        if biz_code == _BIZ_RATE_LIMIT or "rate limit" in biz_msg.lower():
            raise RateLimited(
                f"DeepSeek rate limit reached ({biz_msg or 'no message'}). The "
                "account is sending requests too quickly; wait a minute before "
                "retrying."
            )
        raise RuntimeError(
            f"DeepSeek rejected the request: {biz_msg or biz_code}"
        )

    biz = payload.get("biz_data")
    if biz is None:
        _log.warning("unexpected response shape: %s", data)
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
    ):
        # `allow_interactive=False` makes session resolution non-blocking: it
        # uses a cached/headless session and raises LoginRequired instead of
        # opening a browser window. The server passes False (see server/api.py).
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent server requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._gate = threading.BoundedSemaphore(MAX_CONCURRENCY)
        # Wedge detection (see `wedged`): how many requests hold a gate slot,
        # and when anything last moved (slot taken/released, SSE chunk arrived).
        self._pacer = _Pacer()
        self._inflight = 0
        self._progress_ts = time.time()
        self._state_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            # `pool` bounds how long a request may wait for a free connection —
            # without it a wedged connection pool blocks callers forever.
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=120.0, pool=30.0),
        )

    def _base_headers(self) -> dict:
        # No content-type here: httpx derives it per request (application/json
        # for `json=` bodies, multipart with boundary for `files=` uploads); a
        # fixed client-level value would break the multipart file upload.
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        return _biz(r.json())["chat_session"]["id"]

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        r = self._http.post(
            "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        r.raise_for_status()
        challenge = _biz(r.json())["challenge"]
        with self._pow_lock:
            return self._pow.make_header(challenge)

    # --- public API ---------------------------------------------------------

    def upload_file(self, data: bytes, filename: str = "image.png",
                    mime: str = "image/png") -> str:
        """Upload a file (e.g. an image) and return its DeepSeek file id.

        The id goes into a completion's `ref_file_ids` (see `stream`/`chat`).
        Blocks until DeepSeek has finished parsing the file (it OCRs images),
        because a completion referencing an unparsed file is rejected.
        """
        headers = {"x-ds-pow-response": self._pow_header(UPLOAD_PATH)}
        r = self._http.post(
            UPLOAD_PATH, files={"file": (filename, data, mime)}, headers=headers
        )
        r.raise_for_status()
        file_id = _biz(r.json())["id"]

        deadline = time.time() + FILE_PARSE_TIMEOUT
        while True:
            r = self._http.get(FETCH_FILES_PATH, params={"file_ids": file_id})
            r.raise_for_status()
            files = _biz(r.json()).get("files") or []
            status = files[0].get("status") if files else None
            if status == "SUCCESS":
                return file_id
            if status not in (None, "PENDING", "PARSING"):
                raise RuntimeError(
                    f"DeepSeek could not process the uploaded file "
                    f"{filename!r} (status: {status})"
                )
            if time.time() > deadline:
                raise RuntimeError(
                    f"DeepSeek did not finish parsing {filename!r} within "
                    f"{FILE_PARSE_TIMEOUT:.0f}s (last status: {status})"
                )
            time.sleep(0.5)

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        `ref_file_ids` attaches previously uploaded files (see `upload_file`).
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified). The chat
            # session itself is created lazily on first iteration, inside the
            # concurrency gate.
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type,
                       thinking, search, ref_file_ids)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
    ) -> Reply:
        """Return the complete reply (`.text`) plus its `.conversation_id`."""
        s = self.stream(prompt, conversation_id=conversation_id, model=model,
                        thinking=thinking, search=search, ref_file_ids=ref_file_ids)
        parts = {"thinking": [], "text": []}
        for kind, chunk in s.events():
            parts[kind].append(chunk)
        return Reply(text="".join(parts["text"]),
                     conversation_id=s.conversation_id,
                     thinking="".join(parts["thinking"]) or None)

    def _touch(self, delta: int = 0) -> None:
        """Record upstream progress (and optionally adjust the in-flight count)."""
        with self._state_lock:
            self._inflight += delta
            self._progress_ts = time.time()

    def wedged(self, timeout: float) -> bool:
        """True when every slot is taken and nothing has moved for `timeout` s.

        This is the signature of the httpcore sync-pool deadlock we hit in
        production (an abandoned streaming response finalised by the GC on a
        thread that already held the pool lock, poisoning it forever): requests
        enter, nothing ever completes, and no SSE chunk arrives. A healthy but
        slow upstream keeps producing chunks, so it never trips this.
        """
        with self._state_lock:
            return (
                self._inflight >= MAX_CONCURRENCY
                and time.time() - self._progress_ts > timeout
            )

    def pace_hint(self) -> int:
        """Seconds a caller should wait before retrying, given recent throttling.

        Fed to `Retry-After` so the client's own backoff matches ours instead
        of retrying into a wall we already know is there.
        """
        return max(5, int(self._pacer.gap + 0.999))

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Iterable of reply-text chunks. After it's consumed, `.conversation_id`
    holds the token for resuming the conversation.

    Iterating the stream yields only the reply text. To also see DeepThink
    reasoning, iterate `.events()`, which yields ("thinking"|"text", chunk)
    pairs."""

    def __init__(self, client: "DeepSeekClient", prompt: str,
                 session_id: Optional[str], parent_id: Optional[int],
                 model: Optional[str], thinking: bool, search: bool,
                 ref_file_ids: Optional[list] = None):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._ref_file_ids = list(ref_file_ids or [])
        self._message_id: Optional[int] = None

    def __iter__(self) -> Iterator[str]:
        return (chunk for kind, chunk in self.events() if kind == "text")

    def events(self) -> Iterator[tuple]:
        """Yield ("thinking"|"text", chunk) pairs as they arrive."""
        # Everything that touches DeepSeek happens under the gate, so at most
        # MAX_CONCURRENCY requests are upstream at once and the rest fail fast.
        if not self._client._gate.acquire(timeout=QUEUE_TIMEOUT):
            raise ServerBusy(
                f"All {MAX_CONCURRENCY} upstream slots stayed busy for "
                f"{QUEUE_TIMEOUT:.0f}s; the server is overloaded. Retry shortly."
            )
        self._client._touch(+1)
        try:
            for event in self._run():
                self._client._touch()
                yield event
        finally:
            self._client._touch(-1)
            self._client._gate.release()

    def _attempt(self, meta: dict) -> Iterator[tuple]:
        """One completion request; yields its events and fills `meta`."""
        if self._session_id is None:
            self._session_id = self._client.create_chat_session()
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": self._ref_file_ids,
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model
        # PoW challenges are short-lived, so solve right before the request.
        headers = {"x-ds-pow-response": self._client._pow_header()}
        with self._client._http.stream(
            "POST", COMPLETION_PATH, json=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            yield from _parse_sse(resp.iter_lines(), meta)

    def _log_summary(self, meta: dict, seen: dict, started: float,
                     attempt: int) -> None:
        """Record what upstream actually produced, per fragment kind.

        This is the view that matters when a reply looks wrong: which kinds of
        content arrived and how much of each. A reply that is empty, or that
        came back entirely as thinking, or in a fragment type we did not expect,
        is obvious here and invisible from the finished text alone.
        """
        detail = ", ".join(f"{k}={v}" for k, v in sorted(seen.items())) or "nothing"
        _log.info(
            "<= upstream stream: %s (%.1fs, attempt %d, message_id=%s, "
            "model_type=%s, thinking=%s, search=%s, files=%d, prompt=%d chars)",
            detail, time.time() - started, attempt,
            meta.get("message_id"), self._model or "inherited",
            self._thinking, self._search, len(self._ref_file_ids),
            len(self._prompt),
        )

    def _run(self) -> Iterator[tuple]:
        # DeepSeek answers a throttled request by accepting it and returning an
        # empty stream, almost instantly. That is worth one quiet retry: nothing
        # has been emitted yet, so no output can be duplicated, and the wasted
        # attempt cost the account nothing. Retrying more than once would just
        # be hammering an account that is already being told to slow down.
        meta: dict = {}
        emitted = False
        pacer = self._client._pacer
        pacer.wait()
        started = time.time()
        seen: dict = {}
        try:
            for kind, chunk in self._attempt(meta):
                emitted = True
                seen[kind] = seen.get(kind, 0) + len(chunk)
                yield (kind, chunk)
        except RateLimited:
            # An explicit refusal is the clearest throttle signal there is.
            pacer.on_throttled()
            raise
        self._log_summary(meta, seen, started, attempt=1)
        (pacer.on_success if emitted else pacer.on_throttled)()

        if not emitted and meta.get("message_id") is not None:
            time.sleep(EMPTY_RETRY_DELAY)
            pacer.wait()
            # A fresh session: the first attempt already consumed a message slot
            # in this thread, and nothing was emitted from it, so starting clean
            # keeps the thread history free of a stray empty turn.
            if self._parent_id is None:
                self._session_id = None
            meta = {}
            started, seen = time.time(), {}
            for kind, chunk in self._attempt(meta):
                emitted = True
                seen[kind] = seen.get(kind, 0) + len(chunk)
                yield (kind, chunk)
            self._log_summary(meta, seen, started, attempt=2)
            (pacer.on_success if emitted else pacer.on_throttled)()

        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]
            return
        # HTTP 200 but nothing usable came back. Only an over-long prompt is
        # worth naming as a cause, and only for "expert", which goes silent
        # above roughly 160k characters; "default" handles far more (verified
        # past 2M). Anything else is a transient upstream failure, so say so
        # instead of blaming a length that is very likely fine.
        size = len(self._prompt)
        model = self._model or "inherited"
        if model == "expert" and size > 160_000:
            raise RuntimeError(
                f"DeepSeek returned an empty response for a {size:,}-character "
                "prompt. The expert model stops answering above roughly 160,000 "
                "characters; use the deepseek-chat model or shorten the prompt."
            )
        raise RuntimeError(
            "DeepSeek returned an empty response "
            f"({size:,}-character prompt, model_type={model}"
            f"{', some text received' if emitted else ', no content at all'}). "
            "This is usually a transient upstream failure — retry. If it "
            "persists, your session may have expired: re-run "
            "`python -m deepseek.auth`."
        )

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


# Fragment types, mapped to the event kind callers see.
#
# "READ_LINK" appears whenever the prompt mentions a URL: DeepSeek switches to a
# link-reading mode and puts the WHOLE reply in that fragment — prose and all —
# sometimes without ever emitting a RESPONSE fragment. Treating it as anything
# but reply text silently loses the answer, or truncates one that spans both.
#
# Unknown types therefore default to text as well: DeepSeek adds fragment kinds
# over time, and dropping an unrecognised one costs the user their reply, while
# including it at worst adds some stray text we can see and fix.
#
# "TIP" is the exception, and the reason that default needs a deny-list: it is
# the web UI's own notice bar, not the model speaking. It carries lines like
# "This response is AI-generated, for reference only." (style WARNING), which
# DeepSeek shows beside the answer and would otherwise be appended to it.
_FRAGMENT_KINDS = {"THINK": "thinking", "RESPONSE": "text", "READ_LINK": "text"}
_SKIPPED_FRAGMENTS = {"TIP"}
_DEFAULT_FRAGMENT_KIND = "text"


_reported_fragments: set = set()


def _fragment_kind(frag_type) -> Optional[str]:
    """Event kind for a fragment type, or None when it is not reply content."""
    if frag_type in _SKIPPED_FRAGMENTS:
        return None
    if frag_type not in _FRAGMENT_KINDS:
        # Every fragment type DeepSeek added since this was written has caused a
        # bug that took a session to find (READ_LINK swallowed whole replies;
        # TIP appended a disclaimer). Say so the first time each is seen, so the
        # next one is a log line instead of an investigation.
        if frag_type not in _reported_fragments:
            _reported_fragments.add(frag_type)
            _log.warning(
                "unknown fragment type %r — treating its content as reply "
                "text; check whether that is right", frag_type)
    return _FRAGMENT_KINDS.get(frag_type, _DEFAULT_FRAGMENT_KIND)


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[tuple]:
    """Turn DeepSeek's SSE completion stream into ("thinking"|"text", chunk) deltas.

    The stream sends an initial snapshot frame whose `v` is the full response
    object (with typed `fragments[]`: THINK for DeepThink reasoning, RESPONSE
    for the reply — each may carry initial `content`), then append frames that
    all target the LAST fragment:
      * {"p":"response/fragments/-1/content","o":"APPEND","v":" what"} (sets path)
      * {"v":"'s"}                                                     (appends to it)
    A new fragment mid-stream (thinking done, reply starting) arrives as
      * {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"2",...}]}
    whose `content` is already the reply's first token, so it must be emitted.
    We track the type of the last fragment to label each content delta.

    If `meta` is given, the assistant's `message_id` is recorded into it (used to
    build the resumable conversation_id). The stream opens with

        event: ready
        data: {"request_message_id":1,"response_message_id":2,...}

    which is the authoritative source and always present, so it is read first;
    the snapshot and any `.../message_id` path frame refine it afterwards.
    """
    active_path: Optional[str] = None
    kind: Optional[str] = None  # event kind of the fragment appends target
    snapshot_seen = False

    for line in lines:
        if not line:
            continue
        if not line.startswith("data:"):
            err = _biz_error(line)
            if err:
                raise RuntimeError(f"DeepSeek rejected the request: {err}")
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # The `ready` frame names the assistant message before any content.
        if meta is not None and isinstance(obj.get("response_message_id"), int):
            meta["message_id"] = obj["response_message_id"]

        v = obj.get("v")

        # Snapshot frame: full response object.
        if isinstance(v, dict) and "response" in v:
            if meta is not None:
                _capture_message_id(meta, v)
            fragments = v["response"].get("fragments", [])
            for frag in fragments:
                frag_kind = _fragment_kind(frag.get("type"))
                if frag_kind and frag.get("content") and not snapshot_seen:
                    yield (frag_kind, frag["content"])
            if fragments:
                kind = _fragment_kind(fragments[-1].get("type"))
                active_path = "response/fragments/-1/content"
            snapshot_seen = True
            continue

        # Path-setting frame.
        if "p" in obj:
            active_path = obj["p"]
            if meta is not None and active_path.endswith("message_id") \
                    and isinstance(v, int):
                meta["message_id"] = v
            # New fragment(s) appended: emit their initial content and retarget
            # subsequent content appends at the (new) last fragment's type.
            if active_path.endswith("fragments") and isinstance(v, list):
                for frag in v:
                    if not isinstance(frag, dict):
                        continue
                    kind = _fragment_kind(frag.get("type"))
                    if kind and frag.get("content"):
                        yield (kind, frag["content"])
                continue
            # Content delta ("o" is APPEND, or absent right after a new fragment).
            if obj.get("o") in (None, "APPEND") and isinstance(v, str) \
                    and active_path.endswith("content") and kind:
                yield (kind, v)
            continue

        # Bare append to the current path.
        if isinstance(v, str) and active_path \
                and active_path.endswith("content") and kind:
            yield (kind, v)


def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Best-effort: pull the assistant message_id out of a snapshot frame.

    DeepSeek nests the assistant message under `response`; we check there first,
    then the snapshot root, accepting `message_id` or `id`.
    """
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return
