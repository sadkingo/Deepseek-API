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
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import Session, get_session
from .pow import DeepSeekPow

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# At most this many requests may talk to DeepSeek at once; the rest wait up to
# DEEPSEEK_QUEUE_TIMEOUT seconds for a slot and then fail fast with ServerBusy.
# One web account can't usefully serve more anyway (DeepSeek muted this account
# once already for request spam), and the bound means a burst of retries piles
# up as quick, visible errors instead of an ever-growing queue of stuck threads.
MAX_CONCURRENCY = int(os.getenv("DEEPSEEK_MAX_CONCURRENCY", "4"))
QUEUE_TIMEOUT = float(os.getenv("DEEPSEEK_QUEUE_TIMEOUT", "45"))


class ServerBusy(RuntimeError):
    """All upstream slots stayed occupied for the whole queue timeout."""

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
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str

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


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
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
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
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

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
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
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> Reply:
        """Return the complete reply (`.text`) plus its `.conversation_id`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search)
        text = "".join(s)
        return Reply(text=text, conversation_id=s.conversation_id)

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

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Iterable of reply-text chunks. After it's consumed, `.conversation_id`
    holds the token for resuming the conversation."""

    def __init__(self, client: "DeepSeekClient", prompt: str,
                 session_id: Optional[str], parent_id: Optional[int],
                 model: Optional[str], thinking: bool, search: bool):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._message_id: Optional[int] = None

    def __iter__(self) -> Iterator[str]:
        # Everything that touches DeepSeek happens under the gate, so at most
        # MAX_CONCURRENCY requests are upstream at once and the rest fail fast.
        if not self._client._gate.acquire(timeout=QUEUE_TIMEOUT):
            raise ServerBusy(
                f"All {MAX_CONCURRENCY} upstream slots stayed busy for "
                f"{QUEUE_TIMEOUT:.0f}s; the server is overloaded. Retry shortly."
            )
        self._client._touch(+1)
        try:
            for chunk in self._run():
                self._client._touch()
                yield chunk
        finally:
            self._client._touch(-1)
            self._client._gate.release()

    def _run(self) -> Iterator[str]:
        if self._session_id is None:
            self._session_id = self._client.create_chat_session()
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": [],
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
        meta: dict = {}
        with self._client._http.stream(
            "POST", COMPLETION_PATH, json=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            yield from _parse_sse(resp.iter_lines(), meta)
        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]
        else:
            # No message id and no text: DeepSeek accepted the request (HTTP 200)
            # but produced nothing. The usual cause is an over-long prompt — the
            # "expert" model stops answering above roughly 164k characters, while
            # "default" handles far more — and it reports this by staying silent.
            raise RuntimeError(
                "DeepSeek returned an empty response for a "
                f"{len(self._prompt):,}-character prompt "
                f"(model_type={self._model or 'inherited'}). It is most likely "
                "too long for this model; try the deepseek-chat model or a "
                "shorter prompt."
            )

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[str]:
    """Turn DeepSeek's SSE completion stream into reply-text deltas.

    The stream sends an initial snapshot frame whose `v` is the full response
    object (with `fragments[].content`), then a series of append frames:
      * {"p":"response/fragments/-1/content","o":"APPEND","v":" what"}  (sets path)
      * {"v":"'s"}                                                       (appends to it)
    We track the active append path and emit only RESPONSE-fragment text.

    If `meta` is given, the assistant's `message_id` is recorded into it (used to
    build the resumable conversation_id). The exact field location can vary, so
    we look in a few plausible spots defensively.
    """
    active_path: Optional[str] = None
    emitted_initial = False

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

        v = obj.get("v")

        # Snapshot frame: full response object.
        if isinstance(v, dict) and "response" in v:
            if meta is not None:
                _capture_message_id(meta, v)
            for frag in v["response"].get("fragments", []):
                if frag.get("type") == "RESPONSE" and frag.get("content"):
                    active_path = "response/fragments/-1/content"
                    if not emitted_initial:
                        emitted_initial = True
                        yield frag["content"]
            continue

        # Path-setting append frame.
        if "p" in obj:
            active_path = obj["p"]
            if meta is not None and active_path.endswith("message_id") \
                    and isinstance(v, int):
                meta["message_id"] = v
            if obj.get("o") == "APPEND" and isinstance(v, str) \
                    and active_path.endswith("content"):
                yield v
            continue

        # Bare append to the current path.
        if isinstance(v, str) and active_path and active_path.endswith("content"):
            yield v


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
