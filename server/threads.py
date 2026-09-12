"""Map OpenAI-style stateless histories onto DeepSeek's server-side threads.

OpenAI clients are stateless: every turn they resend the whole `messages` array.
DeepSeek is the opposite — a thread lives on its servers and each turn sends only
the new prompt. Flattening the array into one prompt bridges that gap, but it
hands the model a transcript, which it tends to continue by writing the user's
next turn too ("User: ...").

So instead we remember which DeepSeek thread a history belongs to. A request
whose earlier messages we've seen resumes that thread and sends only its final
message; the model never sees a transcript, and DeepSeek keeps the context.

Matching has to tolerate what frontends do to a history between turns:

- The message being sent is often decorated on the way out — a roleplay client
  appends a "SYSTEM NOTE: ..." paragraph to the newest user message — and comes
  back undecorated in the next request's history. So a turn is stored as
  (prefix, last message) and the last message matches if the stored text equals
  the resent one or merely extends it by whole paragraphs.
- The system prompt shifts between turns (lore entries injected by keyword), so
  only its opening is fingerprinted (see `message_texts`).
- Assistant prose is not fingerprinted at all; only its tool calls are.

The index is saved to disk (`session/threads.json`) so a server restart does
not turn every open conversation into a fresh thread.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# (role, text) pairs, oldest first.
History = Sequence[Tuple[str, str]]


def history_key(history: History) -> str:
    """A stable fingerprint of a conversation prefix."""
    payload = json.dumps([[r, t] for r, t in history], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ThreadCache:
    """Bounded LRU map of history fingerprint -> DeepSeek conversation_id."""

    def __init__(self, max_entries: int = 512) -> None:
        self._entries: "OrderedDict[str, str]" = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            cid = self._entries.get(key)
            if cid is not None:
                self._entries.move_to_end(key)
            return cid

    def put(self, key: str, conversation_id: str) -> None:
        if not conversation_id:
            return
        with self._lock:
            self._entries[key] = conversation_id
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def forget(self, conversation_id: str) -> None:
        """Drop every key pointing at `conversation_id`.

        Used when DeepSeek says a thread is gone: leaving the mapping in place
        would send the next turn straight back to the same dead thread.
        """
        if not conversation_id:
            return
        session = conversation_id.partition(":")[0]
        with self._lock:
            for key in [k for k, v in self._entries.items()
                        if v == conversation_id or v.startswith(session + ":")]:
                self._entries.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _extends(stored: str, current: str) -> bool:
    """Whether `stored` is `current` plus appended paragraphs.

    Texts are in `message_texts` normal form (stripped lines joined by "\n"),
    so the boundary between the original message and an appended note is a
    newline — "Yeah" must not match "Yeah, I need some".
    """
    if stored == current:
        return True
    return (len(stored) > len(current) and stored.startswith(current)
            and stored[len(current)] == "\n")


class TurnIndex:
    """Remembers, per conversation prefix, which DeepSeek thread answered it.

    An entry is (prefix key, last message text, reply fingerprint) -> thread id:
    the state of a thread right after it replied to a request whose history is
    `prefix + last`. Lookups walk a new request's history backwards asking, for
    each earlier message, "did we answer this?" — see `find`.

    Bounded LRU by prefix key, persisted to `path` when given.
    """

    MAX_LAST = 4000  # longest `last` text kept verbatim in an entry

    def __init__(self, max_entries: int = 512, path: Optional[str] = None) -> None:
        # prefix key -> list of [last_text, reply_fp, cid]
        self._entries: "OrderedDict[str, List[list]]" = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        self._load()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text("utf-8"))
            for key, entries in data.get("entries", []):
                self._entries[key] = [list(e) for e in entries]
        except (OSError, ValueError, TypeError):
            self._entries.clear()

    def _save(self) -> None:
        """Write the index atomically; called with the lock held."""
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"entries": list(self._entries.items())}, ensure_ascii=False),
                "utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass  # the in-memory index still works; a restart just forgets

    # ---- reads/writes ------------------------------------------------------
    def get(self, prefix_key: str, last: str, reply_fp: str = "") -> Optional[str]:
        """The thread that answered `prefix + last` with a reply of `reply_fp`."""
        with self._lock:
            entries = self._entries.get(prefix_key)
            if not entries:
                return None
            for stored_last, stored_fp, cid in reversed(entries):
                if stored_fp == reply_fp and _extends(stored_last, last):
                    self._entries.move_to_end(prefix_key)
                    return cid
            return None

    def put(self, prefix_key: str, last: str, reply_fp: str,
            conversation_id: str) -> None:
        if not conversation_id:
            return
        last = last[: self.MAX_LAST]
        with self._lock:
            entries = self._entries.setdefault(prefix_key, [])
            # Same turn answered again (a regeneration): the newest wins.
            entries[:] = [e for e in entries if not (e[0] == last and e[1] == reply_fp)]
            entries.append([last, reply_fp, conversation_id])
            del entries[:-8]  # a handful of branches per prefix is plenty
            self._entries.move_to_end(prefix_key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
            self._save()

    def find(self, history: History) -> Tuple[Optional[str], int]:
        """Locate the thread a resent history belongs to.

        `history` is the request's `message_texts`, newest last. Returns the
        thread id and the index of the first message that thread has NOT seen,
        or (None, len(history) - 1) when nothing matches. Walks backwards so
        one reworded or retried turn costs one extra message, not the thread.
        """
        n = len(history)
        for i in range(n - 2, -1, -1):
            role_next, fp_next = history[i + 1] if i + 1 < n else ("", "")
            reply_fp = fp_next if role_next == "assistant" else ""
            cid = self.get(history_key(history[:i]), history[i][1], reply_fp)
            if cid:
                resume_from = i + 2 if role_next == "assistant" else i + 1
                return cid, min(resume_from, n - 1)
        return None, n - 1

    def remember(self, history: History, reply_fp: str, conversation_id: str) -> None:
        """Record that `history` (a full request) was answered by `conversation_id`."""
        if not history:
            return
        self.put(history_key(history[:-1]), history[-1][1], reply_fp, conversation_id)

    def forget(self, conversation_id: str) -> None:
        """Drop every entry pointing at `conversation_id`'s DeepSeek session."""
        if not conversation_id:
            return
        session = conversation_id.partition(":")[0]
        with self._lock:
            for key in list(self._entries):
                kept = [e for e in self._entries[key]
                        if not (e[2] == conversation_id or e[2].startswith(session + ":"))]
                if kept:
                    self._entries[key] = kept
                else:
                    del self._entries[key]
            self._save()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._entries.values())
