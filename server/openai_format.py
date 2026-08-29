"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from typing import Callable, Iterable, List, Optional, Tuple

from .schemas import ChatMessage


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

# Tag pairs that may wrap a JSON call, as emitted by various model families.
_TAG_OPENERS = (_TOOL_OPEN, "<tool_call>", "<function_calls>", "<invoke>")
_TAG_CLOSERS = (_TOOL_CLOSE, "</tool_call>", "</function_calls>", "</invoke>")

# ReAct: a line starting "Action:", with the arguments on a later
# "Action Input:" line. No closing marker — it runs to the end of the reply.
_REACT_OPEN = re.compile(r"(?:^|\n)[ \t]*Action[ \t]*:", re.IGNORECASE)
_REACT_PARSE = re.compile(
    r"Action[ \t]*:[ \t]*(?P<name>[A-Za-z_][\w.-]*)[ \t]*\n+"
    r"[ \t]*Action[ \t]*Input[ \t]*:[ \t]*(?P<args>.*)",
    re.IGNORECASE | re.DOTALL,
)

# A call written as a bare JSON object, with no wrapper: `{"name": ...`. Only
# treated as an opener when the caller declared tool names to validate against.
_BARE_CALL_OPEN = re.compile(r"""\{\s*["'\u201c\u201d\u2018\u2019]\s*name\s*["'\u201c\u201d\u2018\u2019]\s*:""")

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
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", obj.get("args"))
    args_str = _as_args_string(args)
    return (name, args_str) if args_str is not None else None


def _parse_react_call(raw: str) -> Optional[Tuple[str, str]]:
    """Parse `Action: name` + `Action Input: {json}`."""
    m = _REACT_PARSE.search(raw)
    if not m:
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


def _parse_any_call(raw: str) -> Optional[Tuple[str, str]]:
    """Parse a captured call in whichever format the model used."""
    return _parse_json_call(raw) or _parse_react_call(raw)


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
    if not names or not text or "{" not in text:
        return text, None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        end = _json_span(text, i)
        if end is None:
            continue
        call = _parse_json_call(text[i:end])
        if call and call[0] in names:
            cleaned = (text[:i].rstrip() + "\n" + text[end:].lstrip()).strip()
            return cleaned, call
    return text, None


def serialize_tool_call(name: str, arguments: str) -> str:
    """The in-prompt text form of a tool call.

    Used both when flattening a client-sent assistant message that carries
    tool_calls, and when recording our own emitted call in the thread cache —
    the two must match for thread resumption to fingerprint correctly.
    """
    return (f'{_TOOL_OPEN}{{"name": {json.dumps(name)}, '
            f'"arguments": {arguments or "{}"}}}{_TOOL_CLOSE}')


def tools_preamble(tools: List[dict]) -> str:
    """The prompt block that teaches the model the tool-call protocol."""
    specs = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        specs.append(json.dumps(
            {"name": fn.get("name"), "description": fn.get("description"),
             "parameters": fn.get("parameters")},
            ensure_ascii=False))
    return (
        "[TOOL PROTOCOL — follow exactly]\n"
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
        "- Emit at most ONE <function_call> per reply, then stop immediately; "
        "the result comes back wrapped in <function_result>...</function_result>.\n"
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
        "[/TOOL PROTOCOL]"
    )


def _canonical_text(m: ChatMessage) -> str:
    """A message's text as the model sees it (and as threads fingerprint it).

    Plain messages are their text content verbatim. Tool traffic gets the
    protocol's text form: assistant tool_calls become <function_call> blocks,
    and role="tool" results are wrapped in <function_result> markers.
    """
    if m.role == "tool":
        return f"<function_result>\n{_text_of(m.content)}\n</function_result>"
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
                 names: Optional[set] = None) -> None:
        self.enabled = enabled
        # Declared tool names. Their presence enables bare-JSON openers, and
        # they are what a bare call is validated against before it is accepted.
        self.names = names or set()
        self._bare = False
        self._resolved = False
        self._call: Optional[Tuple[str, str]] = None
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
        if self.strict:
            return best
        m = _REACT_OPEN.search(s)
        if m and (best is None or m.start() < best[0]):
            # Keep "Action:" itself in the buffer — the ReAct parser needs it.
            lead = 1 if s[m.start()] == "\n" else 0
            best = (m.start(), lead, "", False)
        return best

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
            return out.rstrip()
        cut = max(0, len(self._pending) - _TOOL_HOLD)
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
        raw = self._buf.strip()
        call = _parse_any_call(raw)
        if call and self._bare and call[0] not in self.names:
            # A bare `{"name": ...}` is only a call when it names a declared
            # tool; otherwise it is ordinary JSON belonging to the text.
            call = None
        if self.names:
            # Locate the call within the buffer so the text around it survives:
            # a JSON-ish aside the model wrote before it, or a sentence after
            # it. When the first parse already succeeded this just supplies the
            # leftover; when it failed, it also recovers the call itself.
            cleaned, found = find_bare_call(raw, self.names)
            if found is not None:
                self._cleaned = cleaned
                if call is None:
                    call = found
        self._call = call

    def tool_call(self) -> Optional[Tuple[str, str]]:
        """The captured call as (name, arguments-JSON-string), if parseable."""
        if not self.active:
            return None
        self._resolve()
        return self._call

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
        elif self._call is not None:
            return ""  # the buffer is the call; showing it is the bug
        else:
            text = self._opener + self._buf
        for tag in _TAG_OPENERS + _TAG_CLOSERS:
            text = text.replace(tag, "")
        return text.strip()


def extract_tool_call(text: str, strict: bool = False,
                      known_names: Optional[set] = None
                      ) -> Tuple[str, Optional[Tuple[str, str]]]:
    """Non-streaming form: split `text` into (plain text, parsed call or None).

    `strict` recognises only explicit tag openers — use it for reasoning text,
    where a line like "Action: add the dependency" is deliberation, not a call.
    `known_names` enables the last-resort search for a call written as a bare
    JSON object with no wrapper at all (see `find_bare_call`).
    """
    x = ToolCallExtractor(enabled=True, strict=strict, names=known_names)
    plain = x.feed(text) + x.flush()
    call = x.tool_call()
    if x.active and call is None:  # unparseable: hand the raw text back
        plain = (plain + "\n" + x.abandoned_text()).strip()
    if call is None and known_names:
        # Belt and braces: a bare call the opener scan missed (e.g. an unusual
        # key order) can still be recovered from the assembled text.
        plain, call = find_bare_call(plain, known_names)
    return plain.rstrip(), call


def message_texts(messages: List[ChatMessage]) -> List[Tuple[str, str]]:
    """The (role, text) pairs of a request, for thread fingerprinting."""
    return [(m.role, _canonical_text(m)) for m in messages]


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
                        tool_call: Optional[Tuple[str, str]] = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation. `reasoning` (DeepThink text) is
    attached as `message.reasoning_content`, matching the official DeepSeek API.
    `tool_call` is an emulated (name, arguments-json) pair, delivered as
    OpenAI tool_calls with finish_reason "tool_calls".
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    finish = "stop"
    if tool_call:
        message["tool_calls"] = [_tool_call_obj(*tool_call)]
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
                                   names=known_names)
    leak = RoleLeakFilter(enabled=strip_leak)
    collected = []
    reasoning_all = []
    saw_reasoning = False
    reasoning_closed = False

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
                # A call was buffered, but a non-empty reply means it will not
                # be used (see the fallback rule below). `abandoned_text` gives
                # back the surrounding prose with the call removed, so nothing
                # is lost and no markup is shown.
                leftover = think_tool.abandoned_text()
                if leftover:
                    yield frame({"reasoning_content": leftover})
                think_tool = ToolCallExtractor(enabled=False)  # spent
        yield from emit(tool.feed(d))

    call = tool.tool_call()
    if tool.active:
        # Whatever was buffered but is not the call — an unparseable sentinel,
        # or prose the model wrote around the call — belongs in the reply.
        yield from emit(tool.abandoned_text())
    yield from emit(tool.flush())
    tail = leak.flush()
    if tail:
        collected.append(tail)
        yield frame({"content": tail})

    think_call = think_tool.tool_call()
    think_tail = think_tool.flush()
    if think_tail:
        yield frame({"reasoning_content": think_tail})
    # Use a call found in reasoning only when the reply itself is empty: the
    # model reasoned its way to a call and stopped. A call merely *mentioned*
    # while reasoning before a real answer must not be executed.
    if think_call and not call and not "".join(collected).strip():
        call = think_call
    if think_tool.active:
        # Whatever was buffered but is not the call — prose the model wrote
        # around it — is reasoning text, whether or not the call gets used.
        # `abandoned_text` never contains the call, so this is always safe.
        leftover = think_tool.abandoned_text()
        if leftover:
            yield frame({"reasoning_content": leftover})

    if call is None and known_names:
        # Nothing matched a wrapper, so look for a call written as a bare JSON
        # object. This runs after the text has streamed rather than buffering
        # for it: a write_file argument can be hundreds of lines, and holding
        # that back to find out whether it is a call would stall the reply and
        # reorder it. The call still reaches the client, which is what unsticks
        # the turn; the JSON may also remain visible in the text.
        reply_text = "".join(collected)
        _, call = find_bare_call(reply_text, known_names)
        if call is None and not reply_text.strip():
            # Same rule as the reasoning fallback above: only when no reply.
            _, call = find_bare_call("".join(reasoning_all), known_names)

    conversation_id = getattr(stream, "conversation_id", None)
    if not call and not "".join(collected).strip():
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
    if call:
        yield frame({"tool_calls": [dict(_tool_call_obj(*call), index=0)]})
        # Mirror _canonical_text's serialization exactly, so the thread cache
        # key matches the history the client resends next turn.
        text = "".join(collected)
        call_text = serialize_tool_call(*call)
        reply_text = f"{text}\n{call_text}".strip() if text else call_text
        if on_done:
            on_done(reply_text, conversation_id)
        yield frame({}, finish="tool_calls",
                    extra={"conversation_id": conversation_id})
    else:
        if on_done:
            on_done("".join(collected), conversation_id)
        yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
