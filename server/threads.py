"""Map OpenAI-style stateless histories onto DeepSeek's server-side threads.

OpenAI clients are stateless: every turn they resend the whole `messages` array.
DeepSeek is the opposite — a thread lives on its servers and each turn sends only
the new prompt. Flattening the array into one prompt bridges that gap, but it
hands the model a transcript, which it tends to continue by writing the user's
next turn too ("User: ...").

So instead we remember which DeepSeek thread a history belongs to. A request
whose earlier messages we've seen resumes that thread and sends only its final
message; the model never sees a transcript, and DeepSeek keeps the context.

The map is in-memory and per-process: a restart just means the next request
starts a fresh thread, which is correct, only more verbose for one turn.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
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
