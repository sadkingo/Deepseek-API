"""Map OpenAI-style stateless histories onto DeepSeek's server-side threads.

OpenAI clients are stateless: every turn they resend the whole `messages` array.
DeepSeek is the opposite — a thread lives on its servers and each turn sends only
the new prompt. Flattening the array into one prompt bridges that gap, but it
hands the model a transcript, which it tends to continue by writing the user's
next turn too ("User: ...").

So instead we remember which DeepSeek thread a history belongs to, and resume
it with only the messages it has not seen. The hard part is recognising a
conversation in what the client resends, because frontends rewrite histories
between turns:

- the message being sent is decorated on the way out (a roleplay client appends
  a "SYSTEM NOTE: ..." paragraph to the newest user message) and comes back
  undecorated as history;
- the system prompt is rebuilt every turn (lore entries injected by keyword,
  summaries, author's notes) and cannot be matched on;
- once the context window fills, the OLDEST messages are dropped, so the
  history no longer starts where the thread did;
- any earlier message may be edited, a reply may be regenerated ("swiped") and
  either version kept, the assistant's prose is our own output and may come
  back reshaped.

Hashing whole prefixes survives none of that. This module instead keeps every
turn we answered as a record in a tree — each record knows the thread state it
produced (`cid`), the state it was sent from (`parent`), and the *shape* of the
message it carried (one hash per line) — and aligns a resent history against
that tree (`TurnIndex.find`): find the newest history message we recognise,
then walk its ancestors and the history backwards together, checking that
they agree, and score the alignment by how much evidence supports it. The
best-supported thread state wins and the messages after it are resent.

Once a state is found, the turn is classified by what the client is doing to
it (`Match.kind`): continuing the chat, asking the same question again (a
regeneration, answered with DeepSeek's own Regenerate so the chat shows a
second response rather than a second question), or editing a question (a new
branch). When nothing can be resumed and the message is the FIRST turn of a
known chat, the message text decides between an edit of that chat's opening
(similar text: same chat, new branch) and a new conversation that merely
starts from the same greeting (a new chat).

The index is saved to disk (`session/threads.json`) so a server restart does
not turn every open conversation into a fresh thread.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# A history entry: (role, canonical text, assistant tool-call fingerprint).
# Text is the message as the model would see it; the fingerprint is "" except
# for assistant turns that made tool calls (see openai_format.message_texts).
HistoryItem = Tuple[str, str, str]
History = Sequence[HistoryItem]

# Roles whose messages were sent to the thread as prompts and so can be aligned
# with the thread's recorded turns. Assistant turns are our own output and
# system turns move around (author's notes at a depth), so neither anchors.
_ALIGNABLE = ("user", "tool")

_LINE_CAP = 64      # lines hashed per message
_PREFIX_CAP = 8     # earlier messages a root record remembers, for verifying
_SYS_HEAD = 300     # characters of the system prompt that identify it
_REPLY_HEAD = 160   # characters of our reply used to tell branches apart
_TEXT_CAP = 2000    # characters of a turn's message kept for similarity

# A first turn that differs from a chat's recorded opening is an EDIT of that
# opening when the texts are this similar (difflib ratio), else a NEW chat
# that happens to start from the same greeting. Edits are made right after
# reading the reply, so a recently opened chat accepts a looser rewrite.
EDIT_SIMILARITY = float(os.getenv("THREAD_EDIT_SIMILARITY", "0.55"))
RECENT_EDIT_SIMILARITY = float(os.getenv("THREAD_RECENT_EDIT_SIMILARITY", "0.4"))
RECENT_EDIT_WINDOW = float(os.getenv("THREAD_RECENT_EDIT_WINDOW", str(15 * 60)))


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _lines(text: str) -> List[str]:
    """Non-empty lines with whitespace normalised."""
    return [" ".join(l.split()) for l in text.splitlines() if l.strip()]


def message_shape(text: str) -> List[str]:
    """One hash per line: the unit at which frontends append things."""
    return [_h(l) for l in _lines(text)[:_LINE_CAP]]


def message_hash(text: str) -> str:
    return _h("\n".join(_lines(text)))


def system_head(history: History) -> str:
    """Hash of the opening of the leading system message(s)."""
    parts = []
    for role, text, _ in history:
        if role not in ("system", "developer"):
            break
        parts.append(text)
    if not parts:
        return ""
    return _h(" ".join(" ".join(parts).split())[:_SYS_HEAD])


def reply_head(text: str) -> str:
    """Hash of how a reply opens — enough to recognise it when resent."""
    if "<function_call" in text:
        text = text[: text.index("<function_call")]
    return _h(" ".join(text.split())[:_REPLY_HEAD])


def gist(text: str) -> str:
    """A message reduced to what the user wrote: the note lines a frontend
    appended are dropped (they are identical on every turn and would make any
    two short messages look alike), whitespace is normalised, and the result
    is capped."""
    from server.openai_format import split_trailing_notes  # no import cycle
    body, _ = split_trailing_notes(text)
    return " ".join(body.split())[:_TEXT_CAP]


_TOKEN_CAP = 400


def _tokens(text: str) -> List[str]:
    """Lower-cased words, punctuation dropped. Similarity is measured on
    these rather than on characters: two English sentences share most of
    their letters whatever they say, while an edit keeps most of its words."""
    return re.findall(r"\w+", text.lower())[:_TOKEN_CAP]


def similarity(a: str, b: str) -> float:
    """How alike two message gists are, 0..1 (difflib ratio over words)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    m = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    # The cheap upper bound rules most pairs out without the full alignment.
    if m.quick_ratio() < min(EDIT_SIMILARITY, RECENT_EDIT_SIMILARITY):
        return m.quick_ratio()
    return m.ratio()


def edit_threshold(opened_at: float, now: Optional[float] = None) -> float:
    """Similarity a rewritten first turn needs to count as an edit of a chat
    opened at `opened_at`."""
    now = time.time() if now is None else now
    recent = 0 <= now - opened_at < RECENT_EDIT_WINDOW
    return RECENT_EDIT_SIMILARITY if recent else EDIT_SIMILARITY


def compatible(a: List[str], b: List[str]) -> bool:
    """Whether two shapes are the same message, one possibly decorated.

    Equal, or the shorter is a non-empty line-prefix of the longer: a message
    plus an appended note is still that message. "Yeah" against "Yeah, I need
    some" differs within the line, so it does not match.
    """
    if not a or not b:
        return a == b
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return long_[: len(short)] == short


@dataclass
class Turn:
    """One request we answered, and the thread state that answer produced."""
    cid: str                   # "session:message_id" after our reply
    parent: Optional[str]      # the state it was sent from; None = thread root
    role: str                  # role of the message that carried the turn
    shape: List[str]           # line hashes of that message
    reply_fp: str              # tool calls in our reply ("" if none)
    reply_head: str            # hash of our reply's opening
    sys_head: str              # hash of the system prompt's opening
    prefix: List[str] = field(default_factory=list)  # roots: hashes of the
    #   alignable messages that were flattened into the first prompt, oldest
    #   first, so the history before a root can still be checked
    ts: float = 0.0
    account: str = ""          # which DeepSeek account owns the session: a
    #   thread from a previous (muted, replaced) account is dead and never
    #   worth a request
    text: str = ""             # gist of the message, for telling an edit of
    #   it from a different message (records from before this was stored
    #   have "", and are then only recognised by shape)
    user_mid: Optional[int] = None  # DeepSeek's id of the user message
    poisoned: bool = False     # the reply was a refusal (moderation's canned
    #   text or the model declining): nothing is continued from this state

    @property
    def session(self) -> str:
        return self.cid.partition(":")[0]

    @property
    def message_id(self) -> Optional[int]:
        """DeepSeek's id of the assistant message this turn produced."""
        _, _, mid = self.cid.partition(":")
        return int(mid) if mid.isdigit() else None


@dataclass
class Match:
    cid: str
    resume_from: int   # index of the first history message the thread has not
    #   seen; len(history) when it has seen everything (a continue request)
    depth: int         # aligned turns supporting the match
    sys: bool          # the system prompt's opening agreed
    head: bool         # the client kept the reply this state produced
    exhaustive: bool   # thread root and history start met: nothing unchecked
    continues: bool = False  # the client resent this state's own reply last:
    #   it wants that reply continued, not answered
    sibling: bool = False    # `cid` is a bare session id: the turn is a
    #   regeneration or edit of a thread's FIRST turn, whose earlier state we
    #   never had, so it is re-sent in full as a new branch of the same chat
    #   rather than as yet another chat
    # What the client is doing to the matched state:
    #   "append"      a new message after it (the ordinary next turn)
    #   "continue"    carry on the reply it produced
    #   "regenerate"  the same message this state already answered: answer it
    #                 again as another response (`regenerate_of` is that
    #                 answer's message id)
    #   "edit"        a changed version of a message it already answered: a
    #                 new branch from this state
    kind: str = "append"
    regenerate_of: Optional[int] = None
    similarity: float = 0.0  # edits: how alike the new and old message are
    recovering: bool = False  # the state found was a refusal, so the turn is
    #   sent as an edit of the chat's first message instead

    @property
    def label(self) -> str:
        """Short upper-case verb for the log line."""
        return {"append": "RESUME", "continue": "CONTINUE",
                "regenerate": "REGENERATE", "edit": "EDIT"}.get(self.kind, "RESUME")

    def describe(self) -> str:
        if self.recovering:
            why = ["the state this history continues from was a refusal: "
                   "editing the chat's first message with the whole history"]
            if self.depth:
                why.append(f"{self.depth} turn{'s' if self.depth != 1 else ''} aligned")
            return ", ".join(why)
        if self.sibling:
            what = ("regenerated" if self.kind == "regenerate" else "edited")
            why = [f"{what} first turn of a known chat"]
            if self.kind == "edit":
                why.append(f"opening message {self.similarity:.0%} alike")
            if self.depth:
                why.append(f"{self.depth} earlier message{'s' if self.depth != 1 else ''} agree")
        else:
            why = [f"{self.depth} turn{'s' if self.depth != 1 else ''} aligned"]
            if self.kind == "regenerate":
                why.append(f"same question as message {self.regenerate_of} answered")
            elif self.kind == "edit":
                why.append("changed question: new branch")
        if self.continues:
            why.append("continue the reply")
        if self.exhaustive:
            why.append("back to the start")
        if self.sys:
            why.append("same system prompt")
        if self.head:
            why.append("kept reply")
        return ", ".join(why)


def _alignable(history: History) -> List[int]:
    n = len(history)
    return [i for i in range(n - 1)
            if history[i][0] in _ALIGNABLE and history[i][1].strip()]


class TurnIndex:
    """The tree of turns we have answered, and how to find a history in it."""

    def __init__(self, max_turns: int = 2048, path: Optional[str] = None) -> None:
        self._turns: "OrderedDict[str, Turn]" = OrderedDict()  # cid -> Turn, oldest first
        self._by_first: Dict[str, Set[str]] = {}               # first-line hash -> cids
        self._children: Dict[str, Set[str]] = {}               # parent cid -> cids
        # Why the last lookup found nothing to resume, for the log.
        self._last_miss = ""
        # Roots by the hash of the message just before theirs ("" when the
        # thread began with its very first message), for `_find_sibling`.
        self._roots_by_prev: Dict[str, Set[str]] = {}
        self._max = max_turns
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        self._load()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text("utf-8"))
            for d in data.get("turns", []):
                self._index(Turn(**d))
        except (OSError, ValueError, TypeError):
            self._turns.clear()
            self._by_first.clear()
            self._roots_by_prev.clear()
            self._children.clear()

    def _save(self) -> None:
        """Write the index atomically; called with the lock held."""
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"turns": [asdict(t) for t in self._turns.values()]},
                ensure_ascii=False), "utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass  # the in-memory index still works; a restart just forgets

    # ---- bookkeeping ------------------------------------------------------
    @staticmethod
    def _first(shape: List[str]) -> str:
        return shape[0] if shape else ""

    @staticmethod
    def _prev_key(t: Turn) -> str:
        return t.prefix[-1] if t.prefix else ""

    def _index(self, t: Turn) -> None:
        self._turns[t.cid] = t
        self._turns.move_to_end(t.cid)
        self._by_first.setdefault(self._first(t.shape), set()).add(t.cid)
        if t.parent is None:
            self._roots_by_prev.setdefault(self._prev_key(t), set()).add(t.cid)
        else:
            self._children.setdefault(t.parent, set()).add(t.cid)

    @staticmethod
    def _unindex(table: Dict[str, Set[str]], key: str, cid: str) -> None:
        bucket = table.get(key)
        if bucket:
            bucket.discard(cid)
            if not bucket:
                del table[key]

    def _drop(self, cid: str) -> None:
        t = self._turns.pop(cid, None)
        if t is None:
            return
        self._unindex(self._by_first, self._first(t.shape), cid)
        if t.parent is None:
            self._unindex(self._roots_by_prev, self._prev_key(t), cid)
        else:
            self._unindex(self._children, t.parent, cid)

    # ---- writes -------------------------------------------------------------
    def remember(self, history: History, conversation_id: str,
                 parent: Optional[str], reply_fp: str, reply_text: str,
                 account: str = "", user_mid: Optional[int] = None,
                 poisoned: bool = False) -> None:
        """Record that `history` (a whole request) was answered from `parent`
        and left the thread at `conversation_id`, under `account`. `user_mid`
        is DeepSeek's id of the user message that carried the turn; `poisoned`
        says the reply was a refusal, so the state must not be continued."""
        if not conversation_id or ":" not in conversation_id or not history:
            return
        role, text, _ = history[-1]
        prefix: List[str] = []
        if parent is None:
            # A root carries the whole history in its first prompt. Keep the
            # tail of the alignable messages before it so that part of the
            # history can still be verified later (see `_verify`).
            prefix = [message_hash(history[i][1]) for i in _alignable(history)][-_PREFIX_CAP:]
        t = Turn(cid=conversation_id, parent=parent, role=role,
                 shape=message_shape(text), reply_fp=reply_fp,
                 reply_head=reply_head(reply_text), sys_head=system_head(history),
                 prefix=prefix, ts=time.time(), account=account,
                 text=gist(text) if role in _ALIGNABLE else "",
                 user_mid=user_mid, poisoned=poisoned)
        with self._lock:
            # An account is replaced, never revisited (it was muted); its
            # threads are dead and only clutter the index.
            for cid in [c for c, o in self._turns.items() if o.account != account]:
                self._drop(cid)
            self._drop(conversation_id)
            self._index(t)
            while len(self._turns) > self._max:
                self._drop(next(iter(self._turns)))
            self._save()

    def forget(self, conversation_id: str) -> None:
        """Drop everything known about `conversation_id`'s DeepSeek session."""
        if not conversation_id:
            return
        session = conversation_id.partition(":")[0]
        with self._lock:
            for cid in [c for c, t in self._turns.items() if t.session == session]:
                self._drop(cid)
            self._save()

    # ---- reads --------------------------------------------------------------
    def find(self, history: History, account: str = "") -> Optional[Match]:
        """Locate the thread state a resent history continues from.

        Walks the alignable history messages newest-first. For each, every
        recorded turn carrying a compatible message is a candidate; the
        candidate's ancestors are checked against the earlier history
        (`_verify`), and the surviving candidates are ranked by evidence. The
        first position with an acceptable candidate wins: the thread state
        after that message is resumed and everything later is resent.

        When no state can be resumed, `_find_sibling` looks for a thread that
        BEGAN with the message now being regenerated or edited, so the turn
        can at least stay in that chat as a new branch.

        The match is then classified (`Match.kind`) by comparing the message
        being sent with the turns already answered from that state.
        """
        n = len(history)
        if n < 1:
            return None
        positions = _alignable(history)
        sys_h = system_head(history)
        with self._lock:
            self._last_miss = ""
            for idx in range(len(positions) - 1, -1, -1):
                j = positions[idx]
                shape = message_shape(history[j][1])
                cands = self._by_first.get(self._first(shape), ())
                best: Optional[Tuple[tuple, Match]] = None
                for cid in cands:
                    t = self._turns[cid]
                    if t.account != account or t.role != history[j][0] \
                            or not compatible(t.shape, shape):
                        continue
                    # The client's copy of the reply this state produced, when
                    # it is right after the message and not the message being
                    # sent now.
                    k = j + 1
                    has_reply = k < n - 1 and history[k][0] == "assistant"
                    if has_reply and history[k][2] != t.reply_fp:
                        continue   # a different branch: other tool calls
                    head = has_reply and reply_head(history[k][1]) == t.reply_head
                    # The reply sent LAST, with nothing after it, is a request
                    # to continue it ("continue" in SillyTavern and friends).
                    continues = (k == n - 1 and history[k][0] == "assistant"
                                 and reply_head(history[k][1]) == t.reply_head)
                    verified = self._verify(t, history, positions, idx)
                    if verified is None:
                        continue
                    depth, exhaustive = verified
                    sys_ok = bool(t.sys_head) and t.sys_head == sys_h
                    if not (depth >= 2 or sys_ok or exhaustive):
                        # A lone "ok" matching some thread somewhere is not
                        # evidence of anything.
                        continue
                    resume_from = k + 1 if (has_reply or continues) else k
                    m = Match(cid=t.cid, resume_from=min(resume_from, n),
                              depth=depth, sys=sys_ok, head=head,
                              exhaustive=exhaustive, continues=continues)
                    rank = (depth, head or continues, sys_ok, t.ts)
                    if best is None or rank > best[0]:
                        best = (rank, m)
                if best:
                    self._turns.move_to_end(best[1].cid)
                    m = best[1]
                    state = self._turns[m.cid]
                    if state.poisoned and not m.continues:
                        # Continuing from a refusal only gets another one.
                        # Go back to the chat's first message instead: the
                        # whole history as a new branch from the root.
                        return Match(cid=state.session, resume_from=0,
                                     depth=m.depth, sys=m.sys, head=m.head,
                                     exhaustive=m.exhaustive, sibling=True,
                                     kind="edit", recovering=True)
                    return self._classify(m, history)
            m = self._find_sibling(history, positions, sys_h, account)
            if m is None and not self._last_miss:
                self._last_miss = "no known turn in this history"
            return m

    def last_miss(self) -> str:
        """Why the last `find` returned nothing (for the log)."""
        with self._lock:
            return self._last_miss

    def _classify(self, m: Match, history: History) -> Match:
        """Decide what the client is doing to the matched state.

        Only the message being sent can be a repeat or a rewrite of something
        the state already answered, so this looks at the recorded children of
        the state when that message is the only thing new. The same message
        again is a regeneration of the newest such child; a different message
        where a child exists is an edit; anything else is the next turn.
        """
        n = len(history)
        if m.continues:
            m.kind = "continue"
            return m
        if m.resume_from != n - 1:
            return m  # several messages are new: nothing to compare with
        role, text, _ = history[-1]
        if role not in _ALIGNABLE or not text.strip():
            return m
        kids = [self._turns[c] for c in self._children.get(m.cid, ())
                if self._turns[c].role == role]
        if not kids:
            return m
        shape = message_shape(text)
        same = [k for k in kids if compatible(k.shape, shape)]
        if same:
            k = max(same, key=lambda t: t.ts)
            if k.message_id is not None:
                m.kind, m.regenerate_of = "regenerate", k.message_id
            return m
        m.kind = "edit"
        g = gist(text)
        m.similarity = max((similarity(g, k.text) for k in kids if k.text),
                           default=0.0)
        return m

    def _find_sibling(self, history: History, positions: List[int],
                      sys_h: str, account: str) -> Optional[Match]:
        """A thread whose FIRST turn is the message now being sent again.

        Regenerating or editing a thread's opening turn has no earlier state to
        resume: the thread began with that message. The history before the
        message must agree with what the root remembers of it, and the usual
        evidence rule applies. Then the message itself decides:

        - the same message (a regeneration, or a swipe): DeepSeek's own
          Regenerate on that root's reply, so the chat gains a second response
          rather than a second copy of the question;
        - a rewrite of it (similar text: an edit): re-sent in full as a new
          branch of the same chat (a bare session id resumes a session at its
          root);
        - a different message: NOT this chat. Frontends open every chat from
          the same card — same system prompt, same greeting — so a first turn
          that merely shares that prefix is a new conversation and gets a new
          chat, which is what the user sees on DeepSeek's side too.
        """
        n = len(history)
        role, text, _ = history[-1]
        if role not in _ALIGNABLE or not text.strip():
            return None
        prev_key = message_hash(history[positions[-1]][1]) if positions else ""
        shape = message_shape(text)
        g = gist(text)
        now = time.time()
        best: Optional[Tuple[tuple, Match]] = None
        nearest: Optional[Tuple[float, float, str]] = None  # (sim, need, cid)
        for cid in self._roots_by_prev.get(prev_key, ()):
            t = self._turns[cid]
            if t.account != account or t.role != role:
                continue
            # Verify the remembered prefix against the history, newest first.
            depth, i = 0, len(positions) - 1
            ok = True
            for h in reversed(t.prefix):
                if i < 0:
                    break
                if message_hash(history[positions[i]][1]) != h:
                    ok = False
                    break
                depth, i = depth + 1, i - 1
            if not ok:
                continue
            same = compatible(t.shape, shape)      # regenerated, not edited
            sys_ok = bool(t.sys_head) and t.sys_head == sys_h
            exhaustive = i < 0 and len(t.prefix) <= len(positions)
            if not (depth >= 2 or (sys_ok and (depth >= 1 or same))):
                continue
            sim = 1.0 if same else (similarity(g, t.text) if t.text else 0.0)
            if same:
                kind, regen = "regenerate", t.message_id
            else:
                need = edit_threshold(t.ts, now)
                if sim < need:
                    if nearest is None or sim > nearest[0]:
                        nearest = (sim, need, t.cid)
                    continue
                kind, regen = "edit", None
            m = Match(cid=t.session, resume_from=0, depth=depth, sys=sys_ok,
                      head=same, exhaustive=exhaustive, sibling=True,
                      kind=kind, regenerate_of=regen, similarity=sim)
            rank = (same, sim, depth, sys_ok, t.ts)
            if best is None or rank > best[0]:
                best = (rank, m)
        if best is None and nearest is not None:
            sim, need, cid = nearest
            self._last_miss = (
                f"first turn starts like chat {cid} but the opening message "
                f"is only {sim:.0%} alike (an edit needs {need:.0%}): a new "
                "conversation")
        return best[1] if best else None

    def _verify(self, t: Turn, history: History, positions: List[int],
                idx: int) -> Optional[Tuple[int, bool]]:
        """Check `t`'s ancestry against the history before position `idx`.

        Walks parent links and earlier alignable messages together. Any
        disagreement means the thread's state does not reflect this history
        (an earlier message was edited, or this is a different conversation
        with a similar turn) and the candidate is rejected. Returns the number
        of turns that agreed and whether both sides were exhausted together.
        """
        depth = 1
        i = idx - 1
        node = t
        while node.parent is not None and i >= 0:
            parent = self._turns.get(node.parent)
            if parent is None:
                return depth, False  # evicted: nothing more can be checked
            if not compatible(parent.shape, message_shape(history[positions[i]][1])):
                return None
            depth += 1
            node, i = parent, i - 1
        if node.parent is None and i >= 0 and node.prefix:
            # The root was a flattened history; its remembered tail must match
            # the history before it, newest first.
            for h in reversed(node.prefix):
                if i < 0:
                    break
                if message_hash(history[positions[i]][1]) != h:
                    return None
                depth += 1
                i -= 1
        exhaustive = node.parent is None and i < 0
        return depth, exhaustive

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)


class ThreadCache:
    """Bounded LRU map of string key -> value (used for per-thread tool sets)."""

    def __init__(self, max_entries: int = 512) -> None:
        self._entries: "OrderedDict[str, str]" = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            v = self._entries.get(key)
            if v is not None:
                self._entries.move_to_end(key)
            return v

    def put(self, key: str, value: str) -> None:
        if not value:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
