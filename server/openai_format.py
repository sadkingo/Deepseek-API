"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Callable, Iterable, List, Optional, Tuple

from .schemas import ChatMessage


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


# A reply should never contain the *next* turn. If the model writes one anyway,
# it starts a line with a role label, so we cut the reply there.
_LEAKED_TURN = re.compile(r"(?:^|\n)[ \t]*(?:User|Human|System)[ \t]*[:：]", re.IGNORECASE)
# Mid-stream the buffer start is not the reply start, so "^" would misfire on
# text already emitted; streaming matches newlines only and tests the opening
# separately, while nothing has been emitted yet.
_LEAKED_TURN_STREAM = re.compile(r"\n[ \t]*(?:User|Human|System)[ \t]*[:：]", re.IGNORECASE)
_LEADING_TURN = re.compile(r"^[ \t]*(?:User|Human|System)[ \t]*[:：]", re.IGNORECASE)
# A reply may also open by labelling itself; that prefix is just dropped.
_SELF_LABEL = re.compile(r"^[ \t]*(?:Assistant|AI)[ \t]*[:：][ \t]*", re.IGNORECASE)
# Longest label that could straddle a chunk boundary, held back before emitting.
_HOLD = 16


def strip_role_leak(text: str) -> str:
    """Drop a hallucinated next turn, and any label the reply gave itself."""
    text = _SELF_LABEL.sub("", text, count=1)
    m = _LEAKED_TURN.search(text)
    if m:
        text = text[: m.start()]
    return text.rstrip()


class RoleLeakFilter:
    """Streaming form of `strip_role_leak`.

    Text is released as it arrives, minus a short tail held back so a label
    split across two chunks is still recognised. Once a leaked turn is seen
    nothing further is emitted, but the caller must keep draining the upstream
    iterator so the thread's message id still lands.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stopped = False
        self._pending = ""
        self._emitted = False
        self._drop_ws = False

    def _trim_opening(self) -> bool:
        """Handle the reply's own start. True if the whole reply is a leak."""
        if _LEADING_TURN.match(self._pending):
            self._pending = ""
            self.stopped = True
            return True
        unlabelled = _SELF_LABEL.sub("", self._pending, count=1)
        if unlabelled != self._pending:
            # The label may arrive in its own chunk, with the space after it in
            # the next one, so keep eating whitespace until real text shows up.
            self._drop_ws = True
        self._pending = unlabelled
        if self._drop_ws:
            self._pending = self._pending.lstrip()
            if self._pending:
                self._drop_ws = False
        return False

    def feed(self, chunk: str) -> str:
        if not self.enabled:
            return chunk
        if self.stopped:
            return ""
        self._pending += chunk

        # Only meaningful until real text has gone out; a partial label at the
        # very start stays buffered because _HOLD exceeds any label's length.
        if not self._emitted and self._trim_opening():
            return ""

        m = _LEAKED_TURN_STREAM.search(self._pending)
        if m:
            out = self._pending[: m.start()].rstrip()
            self._pending = ""
            self.stopped = True
            self._emitted = self._emitted or bool(out)
            return out

        cut = max(0, len(self._pending) - _HOLD)
        out, self._pending = self._pending[:cut], self._pending[cut:]
        if out:
            self._emitted = True
        return out

    def flush(self) -> str:
        if not self.enabled or self.stopped:
            return ""
        if not self._emitted and self._trim_opening():
            return ""
        out, self._pending = self._pending, ""
        m = _LEAKED_TURN_STREAM.search(out)
        if m:
            out = out[: m.start()]
        return out.rstrip()


def message_texts(messages: List[ChatMessage]) -> List[Tuple[str, str]]:
    """The (role, text) pairs of a request, for thread fingerprinting."""
    return [(m.role, _text_of(m.content)) for m in messages]


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    Message text is sent verbatim: no role labels or any other wrapper text is
    added, so what reaches DeepSeek is exactly what the caller supplied. Several
    messages are joined by a blank line and nothing else.
    """
    texts = [_text_of(m.content) for m in messages]
    return "\n\n".join(t for t in texts if t)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(
    model: str,
    stream: Iterable[str],
    on_done: Optional[Callable[[str, Optional[str]], None]] = None,
    strip_leak: bool = True,
) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    `stream` is the client's stream object; after it's consumed we read its
    `.conversation_id` and attach it to the final chunk. `on_done` receives the
    reply text and that id once the stream ends.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    leak = RoleLeakFilter(enabled=strip_leak)
    collected = []

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})
    for d in stream:
        # Always drain upstream, even after a leak is cut: the thread's message
        # id only arrives once the underlying response is fully consumed.
        if not d:
            continue
        safe = leak.feed(d)
        if safe:
            collected.append(safe)
            yield frame({"content": safe})
    tail = leak.flush()
    if tail:
        collected.append(tail)
        yield frame({"content": tail})

    conversation_id = getattr(stream, "conversation_id", None)
    if on_done:
        on_done("".join(collected), conversation_id)
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
