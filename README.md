# DeepSeek API: a free LLM API powered by DeepSeek

**Using your own DeepSeek account.** No API key, no credits, no paid plan: it turns the free chat at [chat.deepseek.com](https://chat.deepseek.com) into an API you can call from code.

You can use it in two ways:

- 🐍 **As a Python library:** just call `client.chat("Hi")`. Supports streaming and multi-turn conversations.
- 🔌 **As a local OpenAI-compatible API:** runs a server at `http://localhost:8000/v1` that speaks the OpenAI format, so the official `openai` SDK (and any OpenAI-compatible app) works as a drop-in, with `localhost` in place of OpenAI.

You sign in once in a browser with your DeepSeek account; your session is saved and refreshed automatically after that.

> **Unofficial project.** Not affiliated with or endorsed by DeepSeek. It automates the consumer DeepSeek web experience for personal use, so use it responsibly and within DeepSeek's terms.

---

## Table of contents

- [Why use this?](#why-use-this)
- [Requirements](#requirements)
- [Setup (2 minutes)](#setup-2-minutes)
- [Usage 1: In Python (no server)](#usage-1-in-python-no-server)
- [Usage 2: As an OpenAI-compatible server](#usage-2-as-an-openai-compatible-server)
- [Command line](#command-line)
- [Human-check & proof-of-work (automatic)](#human-check--proof-of-work-automatic)
- [Models, DeepThink & web search](#models-deepthink--web-search)
- [Images (vision)](#images-vision)
- [Tool calling](#tool-calling)
- [Using it from Zed](#using-it-from-zed)
- [Concurrency](#concurrency)
- [Rate limiting](#rate-limiting)
- [Project layout](#project-layout)
- [Notes & limitations](#notes--limitations)
- [License](#license)

---

## Why use this?

- **Free:** uses your normal signed-in DeepSeek account, no API billing.
- **Drop-in OpenAI replacement:** point any OpenAI client at `localhost` and it just works.
- **Full DeepSeek toolset:** pick the fast or expert model, and toggle DeepThink reasoning and web search per request.
- **Streaming + conversations:** token-by-token output and multi-turn threads addressed by `conversation_id`.

---

## Requirements

- **Python 3.9+**
- A **DeepSeek account** (the free one you use for [chat.deepseek.com](https://chat.deepseek.com) is fine)
- Works on Windows, macOS, and Linux

---

## Setup (2 minutes)

```bash
# 1. Clone the project
git clone https://github.com/sums001/Deepseek-API.git
cd "Deepseek-API"
```

**2. Create and activate a virtual environment**

On **macOS / Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

On **Windows** (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> On Windows you may need to allow script execution once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In `cmd.exe` activate with `venv\Scripts\activate.bat` instead.

**3. Install dependencies and sign in**

```bash
# Install dependencies
pip install -r requirements.txt

# Install the browser Playwright needs (one-time)
playwright install chromium

# Sign in once: a browser opens, log into your DeepSeek account
python -m deepseek.auth
```

The login window opens so you can sign in by hand and solve the human-check once. After that your session (bearer token + cookies) is saved under `session/` (git-ignored, never shared) and reused on every run — the cached session is refreshed automatically, so your first request works right away.

> The server can also open this window for you on demand the first time it needs a session, so this step is optional for local single-user use.

---

## Usage 1: In Python (no server)

The simplest way if your code is already Python.

```python
from deepseek import DeepSeekClient

client = DeepSeekClient()                # loads your signed-in session

# Get a full reply
reply = client.chat("Say hello in one short sentence.")
print(reply.text)

# Continue the SAME conversation — pass the id back
reply2 = client.chat("And now in French?", conversation_id=reply.conversation_id)
print(reply2.text)

# Stream the answer as it's typed
for chunk in client.stream("Tell me a short joke"):
    print(chunk, end="", flush=True)
```

`chat()` returns the full text plus a `conversation_id`; pass that id back to keep the thread going, or omit it to start fresh. `stream()` yields the reply piece by piece.

👉 More: [examples/01_direct_chat.py](examples/01_direct_chat.py), [02_direct_conversation.py](examples/02_direct_conversation.py), [03_direct_stream.py](examples/03_direct_stream.py)

---

## Usage 2: As an OpenAI-compatible server

Start a local server that speaks the OpenAI API, so existing OpenAI tools and SDKs work unchanged.

```bash
python app.py
# -> DeepSeek OpenAI-compatible API on http://127.0.0.1:8000
```

Then point any OpenAI client at it (the API key is required by the SDK but ignored):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Or call it with plain HTTP / `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Endpoints**

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat (supports `"stream": true`, `"tools"`, and images, plus optional `"conversation_id"`, `"thinking"`, `"search"`) |
| `GET`  | `/v1/models` | Lists the available models |
| `GET`  | `/healthz` | Health check (rate-limit exempt) |

> Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`, or run `uvicorn server.api:app --host 0.0.0.0 --port 8080`.

👉 More: [examples/04_server_http.py](examples/04_server_http.py), [examples/05_server_stream.py](examples/05_server_stream.py), [examples/06_server_openai_sdk.py](examples/06_server_openai_sdk.py)

---

## Command line

```bash
python -m deepseek.auth          # sign in and save the session
```

---

## Human-check & proof-of-work (automatic)

DeepSeek's chat sits behind two gates, both handled for you:

- **AWS WAF human-check:** access needs a signed-in browser session that has
  cleared the "verify you're human" check. `python -m deepseek.auth` opens a real
  browser so you can sign in and solve it once; the resulting token + cookies are
  cached under `session/` and reused on every request.
- **Proof-of-work:** every completion is gated by a PoW challenge. The bridge
  solves it by running DeepSeek's own `sha3_wasm_bg.wasm` module — the same one
  the browser loads — inside a `wasmtime` sandbox, so there's nothing to do on
  your end.

A cached session is reused for ~6 hours and refreshed headlessly from your saved
Chrome profile when possible; only a full expiry sends you back to the browser.

---

## Models, DeepThink & web search

The `model` name selects **which model** answers, and the `-reasoner` ids also
turn on DeepThink for clients that can only pick a model name:

| Model | DeepSeek mode | DeepThink | Notes |
| --- | --- | --- | --- |
| `deepseek-chat` | Instant | off | Fast default model |
| `deepseek-expert` | Expert | off | Stronger, slower |
| `deepseek-reasoner` | Instant | on | Reasoning streamed as `reasoning_content` |
| `deepseek-expert-reasoner` | Expert | on | Reasoning streamed as `reasoning_content` |

DeepThink and web search can also be toggled per request, on any model, with
`thinking: true` / `search: true` in the request body — or via the OpenAI SDK's
`extra_body`:

```python
resp = client.chat.completions.create(
    model="deepseek-expert",
    messages=[{"role": "user", "content": "What changed in the news today?"}],
    extra_body={"thinking": True, "search": True},
)
```

With DeepThink on, the reasoning arrives separately from the reply — streamed
as `delta.reasoning_content` chunks and attached as `message.reasoning_content`
on non-streamed responses, matching the official DeepSeek API — so the reply
text stays clean.

`conversation_id`, `thinking`, and `search` are non-OpenAI extras. A thread's
model is fixed at creation, so `model` can't be combined with `conversation_id`
on resume. Unknown model names return a `404` (no silent fallback). See
[server/config.py](server/config.py).

---

## Images (vision)

Send images the standard OpenAI vision way — as base64 **data URIs** inside a
message's content parts:

```python
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
)
```

Each image is uploaded to DeepSeek, waited on until parsed, and attached to the
message (`ref_file_ids`). Only data URIs are read — plain `http(s)` image URLs
are ignored, since the server won't fetch arbitrary URLs on a caller's behalf.

---

## Tool calling

OpenAI-style function calling works, but it is **emulated** — DeepSeek's web
chat has no tool-call channel, so the server bridges it in the prompt:

1. Your `tools` schemas are injected as a protocol preamble on the first turn.
2. The model replies with `<function_call>{"name": ..., "arguments": {...}}`.
3. The server parses that out and returns a normal OpenAI `tool_calls` response
   with `finish_reason: "tool_calls"` — the markers never reach your client.
4. You send the result back as a `role: "tool"` message; it reaches the model
   wrapped in `<function_result>` markers.

```python
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}],
)
resp.choices[0].message.tool_calls  # -> [ChatCompletionMessageToolCall(...)]
```

Because it is prompt-level, expect these limits:

- **One call per reply.** Parallel tool calls aren't supported; keep
  `parallel_tool_calls: false`.
- **`tool_choice` is accepted but not enforced** — the model decides, so it may
  answer directly when you demanded a call.
- **Reliability is the model's, not the protocol's.** The parser is
  deliberately liberal — it also accepts ReAct (`Action:` / `Action Input:`),
  `<tool_call>` tags, fenced JSON, curly quotes, single quotes, trailing
  commas, Python `True`/`None` literals, and **raw newlines and tabs inside
  JSON strings** (which any file-writing tool hits, since its arguments carry
  source code). A call that fails to parse becomes visible junk in the user's
  chat, so a call that still can't be parsed is returned as ordinary text
  (protocol markers stripped) rather than dropped.
- **Streaming buffers the call.** Arguments must parse as a whole, so the
  `tool_calls` delta arrives in one frame at the end of the stream.
- **Calls made inside DeepThink reasoning are rescued.** With thinking on, the
  model sometimes ends its reasoning with the call and writes no reply; that
  call is used (and removed from the reasoning) rather than left to stall the
  turn — but only when the reply is otherwise empty, so a call merely
  *mentioned* mid-thought is never executed.

---

## Using it from Zed

Add the server as an OpenAI-compatible provider in Zed's `settings.json`
(`zed: open settings`), then pick the models in the Agent panel. Zed shows
DeepThink reasoning in a collapsible "Thinking" block (`interleaved_reasoning`)
and lets you attach images (`images`):

```json
{
  "language_models": {
    "openai_compatible": {
      "DeepSeek Local": {
        "api_url": "http://localhost:8000/v1",
        "available_models": [
          {
            "name": "deepseek-chat",
            "display_name": "DeepSeek Chat",
            "max_tokens": 128000,
            "max_output_tokens": 32000,
            "capabilities": {
              "tools": true,
              "images": true,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": false,
              "max_tokens_parameter": false
            }
          },
          {
            "name": "deepseek-reasoner",
            "display_name": "DeepSeek Thinking",
            "max_tokens": 128000,
            "max_output_tokens": 32000,
            "reasoning_effort": "medium",
            "capabilities": {
              "tools": true,
              "images": true,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": true,
              "max_tokens_parameter": false
            }
          }
        ]
      }
    }
  }
}
```

Repeat the two entries with `deepseek-expert` / `deepseek-expert-reasoner`
(`max_tokens: 36000` — the Expert model rejects prompts above ~160k
*characters*, which is ~36k tokens of ordinary English at ~4.4 chars/token)
for the Expert variants. The Instant model has no such practical limit
(verified with full recall past 2M characters), so `128000` — the model's
official context window — is safe, and you can raise it if you want Zed to
compact later.

> If you see an "empty response … most likely too long" error at a size well
> under these limits, it is **not** a length problem — that message was wrong
> in older versions. See [upstream throttling](#upstream-throttling-503-overloaded_error).

`"tools": true` lets Zed use the models agentically (reading and editing files,
running commands) — see [Tool calling](#tool-calling) for how that is emulated
and its limits. Zed asks for an API key the first time: enter anything
(e.g. `ok`) — the server doesn't check it.

---

## Concurrency

The server bridges a **single** signed-in DeepSeek account behind one shared
client. The PoW solver's `wasmtime` store isn't reentrant, so upstream calls are
**serialized**: parallel HTTP requests queue behind a lock and run one at a time
(see [server/api.py](server/api.py)). This is intentional — throughput is
sequential, not parallel. Keep concurrent in-flight requests low, and please
don't hammer your account.

---

## Rate limiting

On top of serialization, the bridge enforces a self-imposed rate limit with a
dependency-free sliding-window limiter ([server/ratelimit.py](server/ratelimit.py)):
it caps accepted requests **per client IP** and returns a standard `429` +
`Retry-After` when you exceed it. `/healthz` is exempt.

| Env var | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests/minute accepted per client IP |

```bash
RATE_LIMIT_PER_MINUTE=60 python app.py   # raise it
```

**On the client side, use exponential backoff.** Transient `429`s clear if you
retry with growing delays (e.g. 1s, 2s, 4s). The official `openai` SDK does this
automatically and honours `Retry-After`; with plain HTTP, add a few retries
yourself.

### Upstream throttling (`503 overloaded_error`)

Separately from this server's own limit, **DeepSeek throttles the account** when
you send a lot of requests in a short span. It does this by accepting the
request and returning an empty reply almost instantly.

Because that costs the account nothing and usually clears at once, the client
**retries such a request once** (after `DEEPSEEK_EMPTY_RETRY_DELAY`, default 2s)
before giving up — nothing has been emitted at that point, so no output can be
duplicated. If the retry is empty too, the server returns a
`503 overloaded_error` telling you to retry, rather than handing your client a
blank message that would stall an agent.

If you see it repeatedly, slow down — the account is being rate limited, and
hammering it harder is what gets a DeepSeek account muted. Waiting a minute or
two clears it.

---

## Project layout

| Path | What it does |
| --- | --- |
| [deepseek/](deepseek/) | The core library: `DeepSeekClient`, auth/browser sign-in ([auth.py](deepseek/auth.py)), the HTTP driver ([client.py](deepseek/client.py)), and the PoW solver ([pow.py](deepseek/pow.py)) |
| [server/](server/) | The FastAPI OpenAI-compatible server |
| [examples/](examples/) | Runnable examples for every feature ([examples/README.md](examples/README.md)) |
| [app.py](app.py) | Starts the server |

---

## Notes & limitations

- **Sign in once, then reuse.** The cached session refreshes automatically; you only re-sign-in if it fully expires.
- **Be reasonable.** Please use it in moderation, and don't spam or hammer it with automated bulk requests.
- **No real token counts.** `usage` in responses is a rough ~4-chars/token estimate.
- **Most OpenAI params are accepted but ignored** (`temperature`, `top_p`, `max_tokens`, `tool_choice`); only `model`, `messages`, `stream`, `tools`, `conversation_id`, `thinking`, and `search` do anything.
- **Tool calling is emulated in the prompt** (see [Tool calling](#tool-calling)): one call per reply, and no `tool_choice` enforcement.
- **Images must be data URIs.** Vision works (see [Images](#images-vision)), but only base64 `data:` URLs are read; `http(s)` image links are ignored.
- **Your session is private.** Everything in `session/` (cookies + token) stays on your machine and is git-ignored.

## License

Released under the [MIT License](LICENSE). As this is an unofficial project, you remain responsible for complying with DeepSeek's terms of service.

---

