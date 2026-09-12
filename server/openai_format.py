"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
import uuid
from typing import Callable, Iterable, List, Optional, Tuple

from .schemas import ChatMessage

# Shares the upstream logger so this lands in the request log too.
_log = logging.getLogger("deepseek.upstream")


# data-URI images inside OpenAI vision-style `image_url` parts, e.g.
# "data:image/png;base64,iVBOR..." — the only image form Zed and most local
# frontends send. http(s) image URLs are not fetched (we won't make arbitrary
# outbound requests on a caller's behalf).
_DATA_URI = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)

_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
             "image/webp": "webp"}


def message_images(messages: List[ChatMessage]) -> List[Tuple[str, str, bytes]]:
    """All decodable images in `messages`, as (filename, mime, bytes).

    Reads OpenAI vision-style parts: {"type": "image_url",
    "image_url": {"url": "data:image/...;base64,..."}}. Anything that is not a
    base64 data URI is skipped.
    """
    images = []
    for m in messages:
        if not isinstance(m.content, list):
            continue
        for p in m.content:
            if not (isinstance(p, dict) and p.get("type") == "image_url"):
                continue
            url = (p.get("image_url") or {}).get("url", "")
            match = _DATA_URI.match(url)
            if not match:
                continue
            mime, b64 = match.groups()
            try:
                data = base64.b64decode(b64)
            except (ValueError, TypeError):
                continue
            ext = _MIME_EXT.get(mime, "png")
            images.append((f"image_{len(images) + 1}.{ext}", mime, data))
    return images


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
#
# Besides the generic labels, a frontend may prefix the user's messages with a
# name of its own ("sadking: *walks in*", as roleplay clients do), and the model
# then copies THAT label to write the user's lines. `user_labels` finds such a
# prefix in the history and the patterns below are built to include it, with
# the markdown wrappers the model likes to add ("**sadking**:").
_BASE_LABELS = ("User", "Human", "System")
_WRAP = r"(?:\*\*|\*|__|_)?"
_MARKERS = r"</?\s*(?:function_result|tool_result|tool_response)\s*>"


def _label_alt(labels) -> str:
    names = list(_BASE_LABELS) + [l for l in labels if l]
    return "(?:" + "|".join(re.escape(n) for n in names) + ")"


def _label_re(labels) -> str:
    return rf"[ \t]*{_WRAP}[ \t]*{_label_alt(labels)}[ \t]*{_WRAP}[ \t]*[:：]"


class _TurnPatterns:
    """The leak regexes for one set of labels (the base set plus any extras)."""

    def __init__(self, labels=()):
        self.labels = tuple(labels)
        label = _label_re(self.labels)
        self.leaked = re.compile(rf"(?:^|\n){label}|{_MARKERS}", re.IGNORECASE)
        # Mid-stream the buffer start is not the reply start, so "^" would
        # misfire on text already emitted; streaming matches newlines only and
        # tests the opening separately, while nothing has been emitted yet.
        self.leaked_stream = re.compile(rf"\n{label}|{_MARKERS}", re.IGNORECASE)
        self.leading = re.compile(rf"^{label}", re.IGNORECASE)
        # Longest label that could straddle a chunk boundary, held back before
        # emitting: name plus wrappers, spaces and the colon.
        longest = max(len(n) for n in _BASE_LABELS + self.labels)
        self.hold = max(16, longest + 12)


_BASE_PATTERNS = _TurnPatterns()


def _patterns(labels) -> _TurnPatterns:
    return _TurnPatterns(labels) if labels else _BASE_PATTERNS


# A reply may also open by labelling itself; that prefix is just dropped.
_SELF_LABEL = re.compile(r"^[ \t]*(?:Assistant|AI)[ \t]*[:：][ \t]*", re.IGNORECASE)

# What a user-message prefix may look like: a short name on the first line,
# ending in a colon. Nothing with a newline or a colon inside it.
_USER_PREFIX = re.compile(r"^[ \t]*([^\s:：][^\n:：]{0,38}?)[ \t]*[:：](?=\s)")
_MAX_LABEL = 40


def user_labels(messages: List[ChatMessage]) -> Tuple[str, ...]:
    """The name the client prefixes the user's messages with, if any.

    Roleplay frontends send every user turn as "<name>: text" so the model can
    tell speakers apart in a flat transcript — and having learnt that label,
    the model will sometimes write the user's next line itself. Cutting the
    reply there needs the label, so it is read off the history: the prefix
    most of the text-bearing user messages share. A majority rather than all,
    since the opening turn is often an unprefixed "." or "start" that kicks
    the scene off; and at least two must carry it — or, on the first turn, the
    system prompt must name that user too — so a lone "Question: ..." does not
    turn "Question" into a stop label.
    """
    total = 0
    counts: dict = {}
    first_seen: dict = {}
    for m in messages:
        if m.role != "user":
            continue
        text = _text_of(m.content)
        if not re.search(r"\w", text):
            # Empty, or a bare "." / "..." kick-off: says nothing about labels.
            continue
        total += 1
        found = _USER_PREFIX.match(text)
        if not found:
            continue
        label = found.group(1).strip()
        key = label.lower()
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, label)
    if not counts:
        return ()
    key = max(counts, key=counts.get)
    label = first_seen[key]
    if len(label) > _MAX_LABEL or key in {n.lower() for n in _BASE_LABELS}:
        return ()
    if counts[key] * 2 <= total:
        return ()
    if counts[key] < 2 and not _named_in_system(label, messages):
        return ()
    return (label,)


def _named_in_system(label: str, messages: List[ChatMessage]) -> bool:
    """Whether a system message mentions `label` as a word of its own.

    On the very first turn there is only one prefixed user message to go by,
    which alone is too little ("Question: how do I..." must not become a stop
    label). A roleplay client's system prompt, though, names the user's persona
    throughout, so that corroborates the prefix.
    """
    if " " in label:
        return False
    word = re.compile(rf"(?<![\w]){re.escape(label)}(?![\w])")
    return any(m.role == "system" and word.search(_text_of(m.content))
               for m in messages)


def strip_role_leak(text: str, labels=()) -> str:
    """Drop a hallucinated next turn, and any label the reply gave itself.

    `labels` are extra turn labels to cut at, besides User/Human/System — the
    user's name as the client prefixes it (see `user_labels`).
    """
    pats = _patterns(labels)
    text = _SELF_LABEL.sub("", text, count=1)
    m = pats.leaked.search(text)
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

    def __init__(self, enabled: bool = True, labels=()) -> None:
        self.enabled = enabled
        self.stopped = False
        self._pats = _patterns(labels)
        self._pending = ""
        self._emitted = False
        self._drop_ws = False

    def _trim_opening(self) -> bool:
        """Handle the reply's own start. True if the whole reply is a leak."""
        if self._pats.leading.match(self._pending):
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
        # very start stays buffered because the hold exceeds any label's length.
        if not self._emitted and self._trim_opening():
            return ""

        m = self._pats.leaked_stream.search(self._pending)
        if m:
            out = self._pending[: m.start()].rstrip()
            self._pending = ""
            self.stopped = True
            self._emitted = self._emitted or bool(out)
            return out

        cut = max(0, len(self._pending) - self._pats.hold)
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
        m = self._pats.leaked_stream.search(out)
        if m:
            out = out[: m.start()]
        return out.rstrip()


# ---- tool-call emulation -----------------------------------------------
#
# DeepSeek's web API has no tool_calls channel, so OpenAI function calling is
# emulated over plain text: the tool schemas and a calling protocol are put in
# the prompt, the model emits a call, and we translate that back into OpenAI
# tool_calls objects. Tool results come back from the client as role="tool"
# messages and are sent to the model wrapped in <function_result> markers.
#
# We ASK for <function_call>{...}</function_call>, but accept every shape the
# model actually produces. DeepSeek's web chat is heavily tuned toward ReAct
# ("Action:" / "Action Input:") and slips into it under a long system prompt no
# matter what the preamble says, so the parser is deliberately liberal: being
# strict here just turns a usable call into visible junk in the user's chat.

_TOOL_OPEN = "<function_call>"
_TOOL_CLOSE = "</function_call>"

# The model must never write a tool RESULT: results come from the client, and
# a reply containing one means the model stopped calling tools and started
# imagining the whole conversation — inventing "Successfully applied 1 edit"
# for a call that was never made, then "reading" files it never read. Every
# word after the first fabricated result is fiction, so the reply is cut there
# and only the call that came before it is kept. That call is real; the client
# runs it and sends back the true result.
_FABRICATED_RESULT = re.compile(
    r"</?\s*(?:function_result|function_results|tool_result|tool_response|"
    r"observation)\s*>|(?:^|\n)[ \t]*Observation[ \t]*:",
    re.IGNORECASE)


def cut_at_fabricated_result(text: str) -> str:
    """Everything before the model started inventing tool results."""
    m = _FABRICATED_RESULT.search(text or "")
    if not m:
        return text
    _log.warning(
        "the model invented tool results and carried on the conversation with "
        "itself; keeping only what came before %r (%d of %d characters "
        "discarded)", m.group(0).strip(), len(text) - m.start(), len(text))
    return text[:m.start()]

# Tag pairs that may wrap a JSON call, as emitted by various model families.
_TAG_OPENERS = (_TOOL_OPEN, "<tool_call>", "<function_calls>", "<invoke>")
_TAG_CLOSERS = (_TOOL_CLOSE, "</tool_call>", "</function_calls>", "</invoke>")

# ReAct: a line starting "Action:", with the arguments on a later
# "Action Input:" line. No closing marker — it runs to the end of the reply.
# The label pair varies by whatever the model happens to imitate: "Action:" /
# "Action Input:" is classic ReAct, but "Tool:" / "Arguments:" and several
# other spellings turn up just as often. All of them mean the same thing.
_NAME_LABEL = r"(?:Action|Tool(?:[ _]?Name)?|Function(?:[ _]?Name)?|Command)"
_ARGS_LABEL = (r"(?:Action[ \t]*Input|Tool[ \t]*Input|Function[ \t]*Input"
               r"|Arguments?|Parameters?|Params|Input|With)")
_REACT_OPEN = re.compile(r"(?:^|\n)[ \t]*" + _NAME_LABEL + r"[ \t]*:",
                         re.IGNORECASE)
# The label plus the name that follows it, used to tell a real call from prose.
_NAME_AFTER_LABEL = re.compile(
    r"\n?[ \t]*" + _NAME_LABEL + r"[ \t]*:[ \t]*[`\"']?(?P<name>[A-Za-z_][\w.-]*)",
    re.IGNORECASE)
_REACT_PARSE = re.compile(
    _NAME_LABEL + r"[ \t]*:[ \t]*[`\"']?(?P<name>[A-Za-z_][\w.-]*)[`\"']?[ \t]*\n+"
    r"[ \t]*" + _ARGS_LABEL + r"[ \t]*:[ \t]*(?P<args>.*)",
    re.IGNORECASE | re.DOTALL,
)

# A call written as a bare JSON object, with no wrapper: `{"name": ...`. Only
# treated as an opener when the caller declared tool names to validate against.
_BARE_CALL_OPEN = re.compile(
    r"""\{\s*["'\u201c\u201d\u2018\u2019]\s*(?:name|tool|tool_name|function|action)"""
    r"""\s*["'\u201c\u201d\u2018\u2019]\s*:""")

# Longest opener that could straddle a chunk boundary, held back before emitting.
_TOOL_HOLD = max(len(o) for o in _TAG_OPENERS) + 12


def _strip_fence(raw: str) -> str:
    """Drop a ```json ... ``` wrapper the model may have added."""
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
    return raw.strip()


# Typographic quotes the model sometimes produces instead of ASCII ones. JSON
# rejects them, and a call that fails to parse is shown to the user as raw
# protocol text — so they are normalised before parsing.
_SMART_QUOTES = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
})


def _escape_raw_controls(raw: str) -> str:
    """Escape literal control characters that appear inside JSON strings.

    Writing a file means putting source code in an argument, and models very
    often emit it with real newlines and tabs inside the JSON string rather
    than \\n and \\t. JSON forbids that, so the call fails to parse and is shown
    to the user as text — which looks like the model refusing to act. Only
    characters inside string literals are touched; structure is untouched.
    """
    out = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:                    # previous char was a backslash
            out.append(ch)
            escaped = False
        elif ch == "\\":
            out.append(ch)
            escaped = in_string
        elif ch == '"':
            in_string = not in_string
            out.append(ch)
        elif in_string and ch in "\n\r\t\b\f":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t",
                        "\b": "\\b", "\f": "\\f"}[ch])
        elif in_string and ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _loads_lenient(raw: str):
    """json.loads, retried against the near-misses models actually emit.

    Returns the decoded value, or raises the original JSONDecodeError. Repairs
    are tried cumulatively and only ever loosen parsing, so a strictly valid
    document always decodes on the first attempt and is never rewritten.
    """
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(raw)[0]
    except (json.JSONDecodeError, ValueError) as exc:
        first_error = exc

    candidates = []
    fixed = raw.translate(_SMART_QUOTES)          # curly -> straight quotes
    candidates.append(fixed)
    # Raw newlines/tabs inside strings — by far the most common break, because
    # any tool that writes a file carries source code in its arguments.
    fixed = _escape_raw_controls(fixed)
    candidates.append(fixed)
    # Python literals: True/False/None -> true/false/null (outside strings is
    # the common case; a bare word inside a string is left alone by \b anchors
    # only imperfectly, so this is tried after the plainer repairs).
    candidates.append(re.sub(r"\bTrue\b", "true",
                      re.sub(r"\bFalse\b", "false",
                      re.sub(r"\bNone\b", "null", fixed))))
    # Trailing commas before a closing brace/bracket.
    candidates.append(re.sub(r",\s*([}\]])", r"\1", candidates[-1]))
    # Single-quoted strings/keys -> double-quoted (only when there are no
    # double quotes to confuse, so we never corrupt a valid mixed document).
    if '"' not in candidates[-1]:
        candidates.append(candidates[-1].replace("'", '"'))

    for candidate in candidates:
        try:
            return decoder.raw_decode(candidate)[0]
        except (json.JSONDecodeError, ValueError):
            continue
    raise first_error


def _as_args_string(args) -> Optional[str]:
    """Normalise a parsed `arguments` value to a JSON object string."""
    if isinstance(args, str):
        try:  # some models double-encode arguments
            decoded = _loads_lenient(args)
        except (json.JSONDecodeError, ValueError):
            return None
        # Re-serialise so the client always receives strict JSON, even when
        # the model's own encoding needed repair.
        return json.dumps(decoded, ensure_ascii=False)
    if args is None:
        return "{}"
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return None


def _parse_json_call(raw: str) -> Optional[Tuple[str, str]]:
    """Parse `{"name": ..., "arguments": {...}}`, tags/fences tolerated."""
    for close in _TAG_CLOSERS:
        raw = raw.split(close)[0]
    raw = _strip_fence(raw)
    try:
        obj = _loads_lenient(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    name = (obj.get("name") or obj.get("tool") or obj.get("tool_name")
            or obj.get("function") or obj.get("action"))
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", obj.get("args"))
    args_str = _as_args_string(args)
    return (name, args_str) if args_str is not None else None


def _parse_react_call(raw: str, names: Optional[set] = None
                      ) -> Optional[Tuple[str, str]]:
    """Parse a labelled call: `Action: name` + `Action Input: {json}`.

    When the declared tool names are known the parsed name must be one of them.
    The labels ("Tool:", "Function:") are common enough in ordinary prose that
    without that check a sentence could be mistaken for a call.
    """
    m = _REACT_PARSE.search(raw)
    if not m:
        return None
    if names is not None and m.group("name") not in names:
        return None
    args_raw = _strip_fence(m.group("args").strip())
    # The arguments may be followed by more prose (e.g. an "Observation:" the
    # model hallucinated), so decode just the leading JSON value.
    try:
        args = _loads_lenient(args_raw)
    except (json.JSONDecodeError, ValueError):
        if args_raw.lower().startswith("none") or not args_raw:
            args = {}
        else:
            return None
    args_str = _as_args_string(args)
    return (m.group("name"), args_str) if args_str is not None else None


def _parse_any_call(raw: str, names: Optional[set] = None
                    ) -> Optional[Tuple[str, str]]:
    """Parse a captured call in whichever format the model used."""
    return _parse_json_call(raw) or _parse_react_call(raw, names)


def tools_fingerprint(tools: Optional[List[dict]]) -> str:
    """Stable digest of a tool set, to spot when the caller's tools change.

    Covers names, descriptions and parameter schemas, because a renamed
    argument matters to the model just as much as a new tool does. Order is
    normalised so the same tools in a different order are the same set.
    """
    if not tools:
        return ""
    specs = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        specs.append(json.dumps(
            {"name": fn.get("name"), "description": fn.get("description"),
             "parameters": fn.get("parameters")},
            sort_keys=True, ensure_ascii=False))
    return hashlib.sha256("\n".join(sorted(specs)).encode("utf-8")).hexdigest()


def declared_tool_names(tools: Optional[List[dict]]) -> set:
    """The set of function names the caller declared for this request."""
    names = set()
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _span(text: str, start: int, opener: str = "{", closer: str = "}") -> Optional[int]:
    """Index just past the balanced bracket pair beginning at `start`."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _json_span(text: str, start: int) -> Optional[int]:
    """Index just past the balanced {...} beginning at `start`, or None.

    Brace counting that tracks string literals, so braces and quotes inside a
    string argument (file contents, regexes) do not end the object early. Raw
    newlines inside strings are fine here — they are only a problem for the
    JSON decoder, which runs afterwards on the span this returns.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def find_bare_call(text: str, names: set) -> Tuple[str, Optional[Tuple[str, str]]]:
    """Find a call written as a bare JSON object, with no wrapper tags.

    The model sometimes drops the <function_call> markers and simply writes
    {"name": ..., "arguments": {...}}. There is no opener to detect, so instead
    every JSON object in the text is parsed and accepted only when its `name`
    is one of the tools the caller actually declared. That check is what keeps
    this from firing on example JSON or on a JSON file being written.

    Returns the text with the call removed, and the call.
    """
    cleaned, calls = find_bare_calls(text, names)
    return cleaned, (calls[0] if calls else None)


_EXPR_OPEN = re.compile(r"(?<![\w.])(?P<name>[A-Za-z_][\w.-]*)[ \t]*\(")

# At most this many closing brackets may be supplied for a call the model left
# unterminated. A slip drops one or two; anything more means the reply was cut
# off mid-content, and inventing the rest of a file is not a repair.
_MAX_REPAIRED_CLOSERS = 3


def _repair_unterminated(text: str) -> Optional[str]:
    """Close a JSON value the model left open at the very end of its reply.

    Models drop the last brace surprisingly often — the arguments object is
    closed, the call object is not — and one missing character otherwise costs
    the entire call, which then appears as raw JSON in the chat. Only brackets
    (and at most a closing quote) are added; no content is invented.
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None          # malformed rather than merely unfinished
            stack.pop()
    if not stack or len(stack) > _MAX_REPAIRED_CLOSERS:
        return None
    if escaped:
        text = text[:-1]             # a dangling backslash would escape our quote
    if in_string:
        text += '"'
    return text + "".join(reversed(stack))


def find_expr_calls(text: str, names: set) -> Tuple[str, List[Tuple[str, str]]]:
    """Calls written as `tool_name({"arg": 1})`, and the text without them.

    Another shape models reach for — it looks like code, so it is easy to write
    and easy to miss. Only a name that is a declared tool counts, which keeps
    ordinary prose and real code (`foo(bar)`) out of it.
    """
    if not names or not text or "(" not in text:
        return text, []
    calls, out, i = [], [], 0
    while i < len(text):
        m = _EXPR_OPEN.match(text, i)
        if not m or m.group("name") not in names:
            out.append(text[i])
            i += 1
            continue
        end = _span(text, m.end() - 1, "(", ")")
        if end is None:
            out.append(text[i])
            i += 1
            continue
        inner = text[m.end():end - 1].strip()
        if not inner:
            inner = "{}"
        args = _as_args_string(_try_load(inner))
        if args is None:
            out.append(text[i])
            i += 1
            continue
        calls.append((m.group("name"), args))
        i = end
    if not calls:
        return text, []
    cleaned = re.sub(r"[ \t]*\n[ \t]*\n\s*", "\n\n", "".join(out)).strip()
    return cleaned, calls


# Between a tool's name and its arguments, models put decoration: backticks,
# bold markers, a colon, a code fence, a word like "with" or "arguments". This
# is what may appear in that gap — anything else means the two are unrelated.
_CALL_GAP = re.compile(
    r"""^[\s`'"*_:\-]*"""
    r"""(?:(?:with|and|using|arguments?|args|inputs?|parameters?|params|json)"""
    r"""[\s`'"*_:\-]*)*$""",
    re.IGNORECASE)
# The announcement BEFORE the name ("**Calling:**", "Step 2 - run") is allowed
# to carry words; the gap after the name is kept tight. Both are stripped from
# the visible reply when the call is recognised.
_CALL_LABEL = re.compile(
    r"""^[\s`'"*_:\-#>\d.)]*"""
    r"""(?:(?:calling|call|invoke|invoking|running|run|execute|executing|"""
    r"""using|use|tool|function|action|step|next|now|then|first)"""
    r"""[\s`'"*_:\-#>\d.)]*)*$""",
    re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z]*\n?|~~~[a-zA-Z]*\n?")
_WORD = re.compile(r"[A-Za-z_][\w.-]*")


def find_named_calls(text: str, names: set) -> Tuple[str, List[Tuple[str, str]]]:
    """Calls found by their NAME followed by a JSON object.

    A catch-all for the endless ways a model can announce a call —
    "**Calling:** `read_file`" then a fenced object, "Tool: read_file" then
    "Arguments:", or just the name and the object. Rather than teach the parser
    each new decoration, this looks for a declared tool name with a JSON object
    close behind it and nothing but punctuation, a fence, or a linking word in
    between. Requiring the name to be a declared tool is what keeps prose out.
    """
    if not names or not text or "{" not in text:
        return text, []
    calls: List[Tuple[str, str]] = []
    kept: List[str] = []
    cursor = 0
    for m in _WORD.finditer(text):
        if m.start() < cursor or m.group(0) not in names:
            continue
        brace = text.find("{", m.end())
        if brace == -1 or brace - m.end() > 160:
            continue
        if not _CALL_GAP.match(_FENCE.sub(" ", text[m.end():brace])):
            continue
        end = _span(text, brace)
        args = _as_args_string(_try_load(text[brace:end])) if end else None
        if args is None:
            repaired = _repair_unterminated(text[brace:].rstrip())
            args = _as_args_string(_try_load(repaired)) if repaired else None
            end = len(text) if args is not None else None
        if args is None or end is None:
            continue
        # Swallow the label that introduced the name, and any closing fence.
        line_start = text.rfind("\n", 0, m.start()) + 1
        label = _FENCE.sub(" ", text[line_start:m.start()])
        begin = line_start if _CALL_LABEL.match(label) else m.start()
        after = _FENCE.match(text, end) or _FENCE.match(text, end + 1)
        kept.append(text[cursor:begin])
        calls.append((m.group(0), args))
        cursor = after.end() if after else end
    if not calls:
        return text, []
    kept.append(text[cursor:])
    cleaned = re.sub(r"[ \t]*\n[ \t]*\n\s*", "\n\n", "".join(kept)).strip()
    return cleaned, calls


def _try_load(raw: str):
    """Lenient JSON decode that returns None instead of raising."""
    try:
        return _loads_lenient(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def find_bare_calls(text: str, names: set) -> Tuple[str, List[Tuple[str, str]]]:
    """Every bare-JSON call in `text`, and the text with all of them removed.

    Each candidate is bounded by the START OF THE NEXT ONE rather than by
    bracket balance. Models write several calls in a row and routinely drop the
    outer `}` on each, so balance never returns to zero and a scan started at
    the first call runs off through all of them — losing every call but one.
    Bounding first, then repairing inside the window, keeps them separate.
    """
    def _others(t):
        """The other shapes, tried in order of how specific they are."""
        cleaned, expr = find_expr_calls(t, names)
        if expr:
            return cleaned, expr
        return find_named_calls(t, names)

    if not names or not text or "{" not in text:
        return _others(text)

    starts = [m.start() for m in _BARE_CALL_OPEN.finditer(text)]
    if not starts:
        return _others(text)

    calls: List[Tuple[str, str]] = []
    kept: List[str] = []
    cursor = 0
    for n, begin in enumerate(starts):
        if begin < cursor:
            continue                      # already inside a consumed call
        window_end = starts[n + 1] if n + 1 < len(starts) else len(text)
        chunk = text[begin:window_end]

        # Whole, well-formed call inside the window?
        span = _json_span(chunk, 0)
        call = _parse_json_call(chunk[:span]) if span is not None else None
        consumed = begin + span if span is not None else None

        if not (call and call[0] in names):
            # Otherwise close whatever the model left open and try again. The
            # window stops at the next call, so only this one is repaired.
            repaired = _repair_unterminated(chunk.rstrip())
            call = _parse_json_call(repaired) if repaired else None
            consumed = window_end if (call and call[0] in names) else None

        if call and call[0] in names and consumed is not None:
            kept.append(text[cursor:begin])
            calls.append(call)
            cursor = consumed

    kept.append(text[cursor:])
    if not calls:
        return _others(text)
    cleaned = re.sub(r"[ \t]*\n[ \t]*\n\s*", "\n\n", "".join(kept)).strip()
    # A reply can mix shapes; sweep the remainder for the others too.
    cleaned, more = find_expr_calls(cleaned, names)
    cleaned, named = find_named_calls(cleaned, names)
    return cleaned, calls + more + named


def serialize_tool_call(name: str, arguments: str) -> str:
    """The in-prompt text form of a tool call.

    Used both when flattening a client-sent assistant message that carries
    tool_calls, and when recording our own emitted call in the thread cache —
    the two must match for thread resumption to fingerprint correctly.
    """
    args = arguments or "{}"
    try:
        # Re-serialise canonically. A client that reformats the arguments it
        # received (different spacing or key order) would otherwise produce a
        # different fingerprint for the same call, the thread would not be
        # found, and the whole conversation would be replayed as one flat
        # prompt instead of resumed.
        args = json.dumps(json.loads(args), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return (f'{_TOOL_OPEN}{{"name": {json.dumps(name)}, '
            f'"arguments": {args}}}{_TOOL_CLOSE}')


# Argument names that tell us what a tool can do, so the protocol block can
# explain HOW to use it. Derived from the caller's own schemas rather than
# hardcoded per client, since every frontend names its edit tool differently.
_REPLACE_ARGS = {"old", "old_string", "old_text", "search", "find", "replace_with"}
_CONTENT_ARGS = {"content", "contents", "text", "body", "new_string", "new_text",
                 "file_text", "source"}


def _tool_arg_names(tools: List[dict]) -> set:
    """Every parameter name across the declared tools."""
    names = set()
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        props = (fn.get("parameters") or {}).get("properties")
        if isinstance(props, dict):
            names.update(props)
    return names


def _editing_rules(tools: List[dict]) -> List[str]:
    """Advice on using the declared tools, added only where it applies.

    Models reliably describe an edit instead of making one, and when they do
    call an edit tool they get the mechanics wrong in two specific ways: a
    search string retyped from memory that no longer matches the file, and file
    content wrapped in a markdown fence that then lands on disk. Both are worth
    saying explicitly, but only when the tools in play can actually hit them.
    """
    args = _tool_arg_names(tools)
    names = declared_tool_names(tools)
    can_modify = bool(args & (_REPLACE_ARGS | _CONTENT_ARGS)) or any(
        any(verb in n for verb in
            ("write", "create", "edit", "modify", "update", "delete", "remove",
             "move", "rename", "apply", "patch", "terminal", "shell", "bash",
             "command", "exec", "run"))
        for n in names
    )
    rules = []
    if can_modify:
        rules.append(
            "- To change a file, CALL THE TOOL that changes it. Showing the "
            "new code in your reply changes nothing on disk — the user sees a "
            "suggestion, not an edit. Never end a turn having described a "
            "change you could have made."
        )
    if args & _REPLACE_ARGS:
        rules.append(
            "- For a replace-style edit, the text you search for must match the "
            "file EXACTLY — same indentation, same spacing, same line breaks — "
            "and must appear only once. Read the file first and copy the "
            "snippet verbatim from what you read; never retype it from memory "
            "or tidy it up. If it might not be unique, include more "
            "surrounding lines."
        )
    if args & _CONTENT_ARGS:
        rules.append(
            "- Arguments carrying file content take RAW file text: no ``` "
            "fences, no language tag, no line numbers, no commentary. Whatever "
            "you put there is written to disk exactly as given. When replacing "
            "a whole file, include the complete contents, not an excerpt or a "
            "'... rest unchanged ...' placeholder."
        )
    if args & _REPLACE_ARGS and args & _CONTENT_ARGS:
        rules.append(
            "- To change a file that already exists, EDIT it in place. Do not "
            "create a new file with the updated version: that leaves the "
            "original untouched and the user's change never happens."
        )
    if can_modify:
        rules.append(
            "- After the change is made, say in a sentence or two what you "
            "changed and why. Do NOT paste the file's new contents back into "
            "your reply — the file already has them, and the user asked for the "
            "change, not a copy of it."
        )
    if "path" in args or "file_path" in args:
        rules.append(
            "- Use the exact path form that a directory listing showed you. "
            "Paths are anchored to the project root directory and normally "
            "begin with that directory's own name (e.g. 'My Project/src/a.js'). "
            "Copy that shape rather than inventing one, and never drop the "
            "leading directory to 'shorten' a path that was rejected."
        )
    return rules


def tools_preamble(tools: List[dict], changed: bool = False) -> str:
    """The prompt block that teaches the model the tool-call protocol.

    `changed` marks a mid-conversation announcement — tools switched on, or the
    list altered. That needs saying out loud: a thread that began without tools
    has already settled into "I cannot access your files", and will keep
    refusing unless told plainly that its capabilities just changed.
    """
    specs = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        specs.append(json.dumps(
            {"name": fn.get("name"), "description": fn.get("description"),
             "parameters": fn.get("parameters")},
            ensure_ascii=False))
    header = "[TOOL PROTOCOL — follow exactly]\n"
    if changed:
        header += (
            "UPDATE: your capabilities have changed. You NOW have the tools "
            "listed below and can act on this machine with them — anything you "
            "said earlier about being unable to read, write or run things no "
            "longer applies. Use them instead of refusing or asking the user "
            "to paste content.\n"
        )
    return (
        header +
        "These tools are available to you:\n" + "\n".join(specs) + "\n\n"
        "To use one, end your reply with EXACTLY this form, on its own line:\n"
        '<function_call>{"name": "<tool name>", "arguments": <JSON object>}'
        "</function_call>\n\n"
        "Worked example — to list the current directory you would write:\n"
        '<function_call>{"name": "list_directory", "arguments": '
        '{"path": "."}}</function_call>\n\n'
        "Rules:\n"
        "- The <function_call> and </function_call> markers are REQUIRED. A bare "
        "JSON object on its own is not a call and will not run.\n"
        "- Use ONLY this format. Do NOT write 'Action:', 'Action Input:', "
        "'Thought:', 'Observation:', ```json blocks, or any other tool syntax "
        "— those are not understood and the tool will not run.\n"
        "- You may emit several calls in one reply, each fully wrapped in its "
        "own <function_call>...</function_call> markers on its own line — for "
        "example when reading several files. Then stop; every result comes "
        "back wrapped in <function_result>...</function_result>.\n"
        '- "arguments" must be one JSON object valid against that tool\'s '
        "parameters schema.\n"
        "- Never write a <function_result> yourself, never guess a tool's "
        "output, and never mention this protocol or its markers to the user.\n"
        "- When the user asks for a change, CARRY IT OUT with the tools. Do not "
        "describe the change, print the file for them to copy, or ask for "
        "permission first — call the tool that makes it.\n"
        "- Keep going until the task is done: after each result, either make "
        "the next call or give the final answer.\n"
        "- Inside JSON strings, escape newlines as \\n and quotes as \\\" so the "
        "arguments stay valid JSON when they contain source code.\n"
        "- If no tool is needed, just answer normally.\n"
        + "".join(r + "\n" for r in _editing_rules(tools))
        + "[/TOOL PROTOCOL]"
    )


def tools_reminder(tools: List[dict]) -> str:
    """A short protocol refresher, cheap enough to repeat on every turn.

    The full preamble is sent once per thread, but a long agentic session runs
    many turns past it, and by then the model drifts back to plain markdown —
    answering "here is the updated file" with a fenced code block instead of
    calling the edit tool. The schemas do not need repeating (the thread still
    has them); the FORMAT and the obligation to act do.
    """
    if not tools:
        return ""
    names = ", ".join(sorted(declared_tool_names(tools))) or "your tools"
    line = ('<function_call>{"name": "<tool>", "arguments": <JSON>}'
            "</function_call>")
    return (
        "[REMINDER] Tools are still available: " + names + ". To use one, emit "
        + line + " — exactly that form, markers included. To change a file, "
        "call the tool; posting the new contents as a code block does not "
        "change anything. Do not mention this reminder.[/REMINDER]"
    )


# Errors a client returns when a path is not where it expects. The model reads
# "outside the project", concludes the path is too long, and shortens it —
# which is the opposite of the fix, so it loops. Naming the actual cause next
# to the error breaks that loop.
_PATH_REJECTED = re.compile(
    r"outside (?:of )?(?:the )?project"
    r"|not (?:in|inside|within) (?:the )?project"
    r"|path .{0,30}(?:not allowed|not permitted|denied)",
    re.IGNORECASE)

_PATH_HINT = (
    "[HINT] The path was rejected because it is not anchored to the project, "
    "not because it is wrong or too long. Paths must BEGIN with the project "
    "root directory's own name, exactly as it appears in a directory listing "
    "— for example 'My Project/src/file.js', never 'src/file.js' and never an "
    "absolute path. Re-issue the same call with the root directory name "
    "prepended. Do not shorten the path and do not retry it unchanged."
)


def result_hint(text: str) -> str:
    """A correction to attach to a tool result that reports a known mistake."""
    return _PATH_HINT if _PATH_REJECTED.search(text or "") else ""


def _canonical_text(m: ChatMessage) -> str:
    """A message's text as the model sees it (and as threads fingerprint it).

    Plain messages are their text content verbatim. Tool traffic gets the
    protocol's text form: assistant tool_calls become <function_call> blocks,
    and role="tool" results are wrapped in <function_result> markers.
    """
    if m.role == "tool":
        body = _text_of(m.content)
        hint = result_hint(body)
        if hint:
            body = f"{body}\n{hint}"
        return f"<function_result>\n{body}\n</function_result>"
    text = _text_of(m.content)
    if m.tool_calls:
        calls = "\n".join(
            serialize_tool_call(tc["function"].get("name", ""),
                                tc["function"].get("arguments"))
            for tc in m.tool_calls
            if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
        )
        text = f"{text}\n{calls}".strip() if text else calls
    return text


class ToolCallExtractor:
    """Streaming splitter: plain text out, a tool call captured aside.

    Feed reply-text chunks; everything before a call's opener comes back out
    (with a short tail held back in case an opener straddles chunks). From the
    opener on, text is buffered internally; `tool_call()` parses it once the
    stream ends, accepting any of the formats in `_parse_any_call`. This runs
    BEFORE the role-leak filter so file contents inside tool arguments are
    never mangled by it.
    """

    def __init__(self, enabled: bool, strict: bool = False,
                 names: Optional[set] = None,
                 drop_unparsed: bool = False) -> None:
        self.enabled = enabled
        # Declared tool names. Their presence enables bare-JSON openers, and
        # they are what a bare call is validated against before it is accepted.
        self.names = names or set()
        self._bare = False
        # For the reasoning channel: a half-written call (the model began one
        # mid-thought and abandoned it) is protocol debris. In the reply,
        # handing it back is right — losing a reply is worse than showing an
        # odd one — but reasoning has no such stake, and the debris is exactly
        # what the user ends up staring at.
        self.drop_unparsed = drop_unparsed
        self._resolved = False
        self._json_ok = False
        self._calls: List[Tuple[str, str]] = []
        self._cleaned: Optional[str] = None
        # `strict` recognises only the explicit tag openers, never ReAct.
        # Reasoning text is scanned in strict mode: deliberation routinely
        # writes lines like "Action: add the dependency", and treating those as
        # a call buffers the rest of the reasoning until the stream ends, which
        # reorders it after the reply instead of streaming it in place.
        self.strict = strict
        self.active = False
        self._pending = ""
        self._buf = ""
        self._opener = ""  # opener text, re-prepended if parsing fails

    def _find_opener(self, s: str):
        """Earliest opener in `s` as (index, consumed_len, opener_text, is_bare)."""
        best = None
        for op in _TAG_OPENERS:
            i = s.find(op)
            if i != -1 and (best is None or i < best[0]):
                best = (i, len(op), op, False)
        if self.names:
            m = _BARE_CALL_OPEN.search(s)
            if m and (best is None or m.start() < best[0]):
                # Keep the "{" — the JSON parser needs it.
                best = (m.start(), 0, "", True)
        if self.names:
            for m in _EXPR_OPEN.finditer(s):
                if best is not None and m.start() >= best[0]:
                    break
                if m.group("name") not in self.names:
                    continue
                best = (m.start(), 0, "", False)
                break
        if self.names:
            # A declared tool name announced ahead of a JSON object, however
            # the model dressed it up ("**Calling:** `read_file`" + a fenced
            # object). Matches what find_named_calls accepts, so the call is
            # withheld rather than streamed out and only rescued afterwards.
            for m in _WORD.finditer(s):
                if best is not None and m.start() >= best[0]:
                    break
                if m.group(0) not in self.names:
                    continue
                brace = s.find("{", m.end())
                if brace == -1 or brace - m.end() > 160:
                    continue
                if not _CALL_GAP.match(_FENCE.sub(" ", s[m.end():brace])):
                    continue
                line_start = s.rfind("\n", 0, m.start()) + 1
                label = _FENCE.sub(" ", s[line_start:m.start()])
                start = line_start if _CALL_LABEL.match(label) else m.start()
                best = (start, 0, "", False)
                break
        for m in _REACT_OPEN.finditer(s):
            if best is not None and m.start() >= best[0]:
                break
            if self.names is not None and self.names:
                # Only a label followed by a DECLARED tool name counts. Prose
                # says "Tool: we should pick a better one" often enough that
                # matching the label alone would swallow the rest of the text.
                after = _NAME_AFTER_LABEL.match(s, m.start())
                if not after or after.group("name") not in self.names:
                    continue
            # Keep the label itself in the buffer — the parser needs it.
            lead = 1 if s[m.start()] == "\n" else 0
            best = (m.start(), lead, "", False)
            break
        return best

    def _label_hold(self, s: str) -> Optional[int]:
        """Where to stop emitting because a label line is still arriving.

        `Tool:` may be in hand before the tool name is, and the name is what
        decides whether this is a call. Emitting the label and then discovering
        the call would leave the label on screen, so text is held from the
        label until its line completes.
        """
        if not self.names:
            return None
        holds = []
        last = None
        for m in _REACT_OPEN.finditer(s):
            last = m
        if last and "\n" not in s[last.end():]:
            holds.append(last.start())
        # An expression call whose closing bracket has not arrived yet: emitting
        # `read_file(` and only then recognising the call would leave it on
        # screen, so hold from the name.
        for m in _EXPR_OPEN.finditer(s):
            if m.group("name") in self.names and _span(s, m.end() - 1, "(", ")") is None:
                holds.append(m.start())
                break
        # A declared name near the end whose object has not arrived yet: hold
        # from its line, or the name would be emitted and the call recognised
        # only after it was already on screen.
        for m in _WORD.finditer(s):
            if m.group(0) not in self.names:
                continue
            rest = s[m.end():]
            brace = rest.find("{")
            if brace == -1:
                if _CALL_GAP.match(_FENCE.sub(" ", rest)) and len(rest) <= 160:
                    holds.append(max(0, s.rfind("\n", 0, m.start()) + 1))
                continue
            if _CALL_GAP.match(_FENCE.sub(" ", rest[:brace])) \
                    and _span(s, m.end() + brace) is None:
                holds.append(max(0, s.rfind("\n", 0, m.start()) + 1))
        return min(holds) if holds else None

    def feed(self, chunk: str) -> str:
        if not self.enabled:
            return chunk
        if self.active:
            self._buf += chunk
            return ""
        self._pending += chunk
        found = self._find_opener(self._pending)
        if found:
            i, consumed, opener, is_bare = found
            out = self._pending[:i]
            self._buf = self._pending[i + consumed:]
            self._pending = ""
            self.active = True
            self._opener = opener
            self._bare = is_bare
            self._label = not opener and not is_bare
            return out.rstrip()
        cut = max(0, len(self._pending) - _TOOL_HOLD)
        hold = self._label_hold(self._pending)
        if hold is not None:
            cut = min(cut, hold)
        out, self._pending = self._pending[:cut], self._pending[cut:]
        return out

    def flush(self) -> str:
        """Remaining plain text (empty once a call started)."""
        out, self._pending = self._pending, ""
        return out

    def _resolve(self) -> None:
        """Work out, once, what the buffer holds: a call, some text, or both."""
        if self._resolved:
            return
        self._resolved = True
        raw = cut_at_fabricated_result(self._buf).strip()
        calls: List[Tuple[str, str]] = []
        if self.names:
            # Collect EVERY call in the buffer, and keep the prose around them:
            # a JSON-ish aside before the first, a sentence after the last, and
            # the batch of calls models write when asked to read several files.
            cleaned, calls = find_bare_calls(raw, self.names)
            if calls:
                self._cleaned = cleaned
        if not calls and self.names:
            cleaned, named = find_named_calls(raw, self.names)
            if named:
                self._cleaned, calls = cleaned, named
        if not calls:
            # Tag- or label-shaped call, which is not bare JSON to scan for.
            one = _parse_any_call(raw, self.names or None)
            if one and not (self._bare and one[0] not in self.names):
                calls = [one]
        self._calls = calls
        # Did the buffer at least contain a complete JSON object? Valid JSON
        # that names no declared tool is ordinary content and must be kept;
        # a half-written object is debris and may be dropped.
        self._json_ok = False
        if raw.startswith("{"):
            end = _json_span(raw, 0)
            if end is not None:
                try:
                    _loads_lenient(raw[:end])
                    self._json_ok = True
                except (json.JSONDecodeError, ValueError):
                    pass

    def tool_calls(self) -> List[Tuple[str, str]]:
        """Every captured call, as (name, arguments-JSON-string) pairs."""
        if not self.active:
            return []
        self._resolve()
        return self._calls

    def tool_call(self) -> Optional[Tuple[str, str]]:
        """The first captured call, or None."""
        calls = self.tool_calls()
        return calls[0] if calls else None

    def abandoned_text(self) -> str:
        """The buffered text that is NOT part of a call.

        Content is handed back rather than dropped — losing a reply is worse
        than showing an odd one — but never the call itself: an empty string
        means the buffer held nothing but markup. Protocol tags are stripped so
        the user sees the model's text, not our markers.
        """
        if not self.active:
            return ""
        self._resolve()
        if self._cleaned is not None:
            text = self._cleaned
        elif self._calls:
            return ""  # the buffer is the call; showing it is the bug
        elif self.drop_unparsed and (self._bare or self._opener) \
                and not self._json_ok and not self._label:
            return ""  # an unfinished call in the reasoning: not worth showing
        else:
            text = self._opener + self._buf
        for tag in _TAG_OPENERS + _TAG_CLOSERS:
            text = text.replace(tag, "")
        return text.strip()


def extract_tool_calls(text: str, strict: bool = False,
                       known_names: Optional[set] = None
                       ) -> Tuple[str, List[Tuple[str, str]]]:
    """Non-streaming form: split `text` into (plain text, parsed call or None).

    `strict` recognises only explicit tag openers — use it for reasoning text,
    where a line like "Action: add the dependency" is deliberation, not a call.
    `known_names` enables the last-resort search for a call written as a bare
    JSON object with no wrapper at all (see `find_bare_call`).
    """
    x = ToolCallExtractor(enabled=True, strict=strict, names=known_names)
    plain = x.feed(text) + x.flush()
    calls = x.tool_calls()
    if x.active:
        # Prose around the calls (or the raw text when nothing parsed).
        leftover = x.abandoned_text()
        if leftover:
            plain = (plain + "\n" + leftover).strip()
    plain = cut_at_fabricated_result(plain)
    if not calls and known_names:
        # Belt and braces: calls the opener scan missed (an unusual key order,
        # say) can still be recovered from the assembled text.
        plain, calls = find_bare_calls(plain, known_names)
    return plain.rstrip(), calls


def assistant_fingerprint(tool_calls) -> str:
    """What an assistant turn contributes to a history fingerprint: its calls.

    NOT its prose. The assistant text is our own output coming back to us, and
    the client is free to reshape it — Zed returns assistant turns as a list of
    parts that includes the model's *reasoning* alongside the reply, so the text
    it sends back is not the text we produced. Fingerprinting on that meant the
    prefix never matched, the thread was never found, and every turn replayed
    the entire conversation as one flat prompt (186,000 characters in the case
    this was written for). The tool calls are stable on both sides and are what
    actually distinguishes one branch of a conversation from another.
    """
    if not tool_calls:
        return ""
    return "\n".join(
        serialize_tool_call(tc["function"].get("name", ""),
                            tc["function"].get("arguments"))
        for tc in tool_calls
        if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
    )


def message_texts(messages: List[ChatMessage]) -> List[Tuple[str, str, str]]:
    """The (role, text, fingerprint) triples of a request, for thread matching.

    Text is the message as the model sees it (`_canonical_text`); the
    fingerprint is the assistant turn's tool calls (`assistant_fingerprint`),
    "" for every other role. How these are compared — which roles anchor, how
    much of a system prompt counts, what an appended note looks like — is the
    business of `threads.TurnIndex`.
    """
    return [(m.role, _canonical_text(m),
             assistant_fingerprint(m.tool_calls) if m.role == "assistant" else "")
            for m in messages]


def flatten_directive(messages: List[ChatMessage]) -> str:
    """Closing instruction for a history replayed as one flat prompt.

    Without a thread to resume, the model receives the entire conversation —
    tool results, its own past replies — as a single blob with no turn
    structure. It then imitates what it sees: past replies that *describe*
    changes in prose, so the new one describes a change too, and nothing is
    called. Ending with the pending request, plainly stated, gives it a
    present-tense instruction instead of a pattern to copy.
    """
    has_tools = any(m.role == "tool" or m.tool_calls for m in messages)
    if not has_tools:
        return ""
    last_user = next((_text_of(m.content) for m in reversed(messages)
                      if m.role == "user" and _text_of(m.content).strip()), "")
    parts = [
        "[NOW] Everything above is the conversation so far, including tool "
        "results you already received. It is history, not a template — do not "
        "summarise it or repeat a previous answer.",
    ]
    if last_user:
        parts.append("The request still to be carried out is: "
                     + " ".join(last_user.split())[:400])
    parts.append("Continue the work now with the tools. If it needs a change "
                 "to a file, make that change with a tool call — do not "
                 "describe it or paste the file.[/NOW]")
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    Message text is sent verbatim: no role labels or any other wrapper text is
    added, so what reaches DeepSeek is exactly what the caller supplied (tool
    calls/results appear in their protocol text form). Several messages are
    joined by a blank line and nothing else.
    """
    texts = [_canonical_text(m) for m in messages]
    return "\n\n".join(t for t in texts if t)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def _tool_call_obj(name: str, arguments: str) -> dict:
    """An OpenAI tool_calls entry for an emulated call."""
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None,
                        reasoning: str = None,
                        tool_calls: Optional[List[Tuple[str, str]]] = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation. `reasoning` (DeepThink text) is
    attached as `message.reasoning_content`, matching the official DeepSeek API.
    `tool_calls` are emulated (name, arguments-json) pairs, delivered as OpenAI
    tool_calls with finish_reason "tool_calls".
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = [_tool_call_obj(*c) for c in tool_calls]
        finish = "tool_calls"
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
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
    stream,
    on_done: Optional[Callable[[str, Optional[str]], None]] = None,
    strip_leak: bool = True,
    tools_enabled: bool = False,
    known_names: Optional[set] = None,
    on_turn: Optional[Callable[[list, str], None]] = None,
    leak_labels=(),
) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    `stream` is the client's stream object; its `.events()` yields
    ("thinking"|"text", chunk) pairs. Thinking chunks go out as
    `delta.reasoning_content` (the field DeepSeek's official API and clients
    like Zed use); text chunks as `delta.content`. After the stream is consumed
    we read its `.conversation_id` and attach it to the final chunk. `on_done`
    receives the reply text (thinking excluded) and that id once the stream ends.

    With `tools_enabled`, an emulated <function_call> in the reply is captured
    and delivered as an OpenAI tool_calls delta with finish_reason "tool_calls"
    (buffered until the stream ends, since arguments must parse as a whole);
    `on_done` then receives the call's canonical text form so the thread cache
    matches the history the client sends back.

    `leak_labels` are extra turn labels the leak filter cuts at — the user's
    name as the client prefixes it (see `user_labels`).
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

    # Tool extraction runs BEFORE the leak filter: tool arguments may embed
    # file contents with lines like "User: ..." that must not be cut.
    tool = ToolCallExtractor(enabled=tools_enabled, names=known_names)
    # With DeepThink on, the model sometimes ends its *reasoning* with the call
    # and never writes a reply, which would strand the turn: the call would be
    # rendered as thinking text and nothing would run. So reasoning is scanned
    # too — in strict mode, so ordinary deliberation is not mistaken for a call
    # — and that call is used only as a fallback (see below).
    think_tool = ToolCallExtractor(enabled=tools_enabled, strict=True,
                                   names=known_names, drop_unparsed=True)
    leak = RoleLeakFilter(enabled=strip_leak, labels=leak_labels)
    collected = []
    reasoning_all = []
    saw_reasoning = False
    reasoning_closed = False
    held_think_calls: List[Tuple[str, str]] = []

    def emit(text: str):
        safe = leak.feed(text)
        if safe:
            collected.append(safe)
            yield frame({"content": safe})

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})
    for kind, d in stream.events():
        # Always drain upstream, even after a leak is cut: the thread's message
        # id only arrives once the underlying response is fully consumed.
        if not d:
            continue
        if kind == "thinking":
            # Reasoning is a separate channel: no leak filtering, and it never
            # counts as reply text.
            saw_reasoning = True
            reasoning_all.append(d)
            safe = think_tool.feed(d)
            if safe:
                yield frame({"reasoning_content": safe})
            continue
        if saw_reasoning and not reasoning_closed and d.strip():
            # Real reply text is arriving, so reasoning is over. Release what
            # the extractor is holding now, while it still belongs in the
            # reasoning block; flushing it at the end would surface it as a
            # second, stray "thinking" chunk below the finished reply.
            reasoning_closed = True
            tail_think = think_tool.flush()
            if tail_think:
                yield frame({"reasoning_content": tail_think})
            if think_tool.active:
                # `abandoned_text` gives back the surrounding prose with the
                # call removed, so nothing is lost and no markup is shown.
                leftover = think_tool.abandoned_text()
                if leftover:
                    yield frame({"reasoning_content": leftover})
                # Retire the extractor, but KEEP whatever calls it found: the
                # model plans the call while thinking and then writes prose,
                # and those calls are the turn's real intent.
                held_think_calls = think_tool.tool_calls()
                think_tool = ToolCallExtractor(enabled=False)  # spent
        yield from emit(tool.feed(d))

    calls = tool.tool_calls()
    if tool.active:
        # Whatever was buffered but is not the call — an unparseable sentinel,
        # or prose the model wrote around the call — belongs in the reply.
        yield from emit(tool.abandoned_text())
    yield from emit(tool.flush())
    tail = leak.flush()
    if tail:
        collected.append(tail)
        yield frame({"content": tail})

    think_calls = think_tool.tool_calls() or held_think_calls
    think_tail = think_tool.flush()
    if think_tail:
        yield frame({"reasoning_content": think_tail})
    # Use calls found in reasoning when the reply made none of its own. The
    # reply used to have to be empty as well, on the theory that a call in
    # reasoning before a written answer was only being contemplated. In
    # practice the model plans the call while thinking and then writes prose
    # about it, and refusing to run it is exactly the "said it would, didn't"
    # complaint. A reply that makes its own call still wins.
    if think_calls and not calls:
        calls = think_calls
    if think_tool.active:
        # Whatever was buffered but is not the call — prose the model wrote
        # around it — is reasoning text, whether or not the call gets used.
        # `abandoned_text` never contains the call, so this is always safe.
        leftover = think_tool.abandoned_text()
        if leftover:
            yield frame({"reasoning_content": leftover})

    if not calls and known_names:
        # Last resort for a call the opener scan missed (an unusual key order,
        # say): search the assembled text, then the reasoning.
        reply_text = cut_at_fabricated_result("".join(collected))
        _, calls = find_bare_calls(reply_text, known_names)
        if not calls and not reply_text.strip():
            # Same rule as the reasoning fallback above: only when no reply.
            _, calls = find_bare_calls(
                cut_at_fabricated_result("".join(reasoning_all)), known_names)

    conversation_id = getattr(stream, "conversation_id", None)
    if not calls and not "".join(collected).strip():
        # Nothing actionable came back — see the matching note in server/api.py.
        # Report it as an error frame so the client can retry, instead of
        # closing a well-formed but empty turn.
        yield ("data: " + json.dumps({"error": {
            "message": "DeepSeek returned no reply"
                       + (" (it produced only reasoning)" if saw_reasoning else "")
                       + ". This is usually transient, or the account being "
                         "throttled for sending too many requests; wait a "
                         "moment and retry.",
            "type": "overloaded_error"}}) + "\n\n")
        yield "data: [DONE]\n\n"
        return
    if on_turn:
        on_turn(calls, "".join(collected))
    if calls:
        yield frame({"tool_calls": [dict(_tool_call_obj(*c), index=n)
                                    for n, c in enumerate(calls)]})
        # Mirror _canonical_text's serialization exactly, so the thread cache
        # key matches the history the client resends next turn.
        text = "".join(collected)
        call_text = "\n".join(serialize_tool_call(*c) for c in calls)
        reply_text = f"{text}\n{call_text}".strip() if text else call_text
        if on_done:
            on_done(reply_text, conversation_id, calls)
        yield frame({}, finish="tool_calls",
                    extra={"conversation_id": conversation_id})
    else:
        if on_done:
            on_done("".join(collected), conversation_id, [])
        yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
