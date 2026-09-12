# Detection Mitigation & Anti-Detection Engineering Plan

This document provides a comprehensive technical blueprint detailing all detection vectors present in `Deepseek-API`, the underlying mechanics of edge WAF and backend bot detection, architectural countermeasures, and an actionable, phased implementation plan.

---

## 1. Executive Summary & Forensic Analysis

`Deepseek-API` bridges the consumer web interface (`chat.deepseek.com`) into an OpenAI-compatible API. While functional, automating a browser-facing web application exposes several distinct layers of security scrutiny:

```mermaid
graph TD
    A[Client Request] --> B[Layer 1: Network & IP Intelligence]
    B -->|ASN / Data Center Check| C[Layer 2: Edge WAF - Cloudflare & AWS WAF]
    C -->|TLS JA4, HTTP/2 Frames, WAF Token| D[Layer 3: Application PoW Challenge]
    D -->|sha3_wasm Execution & Latency| E[Layer 4: Request Telemetry & Topology]
    E -->|History Pre-flights & Navigation| F[Layer 5: Backend NLP Moderation Classifiers]
    F -->|Policy Check: User Muted / Account Ban| G[Model Inference]
```

### Critical Distinction: Bot Detection vs. Account Moderation Mutes
During deep forensic analysis of request logs and account bans, two separate enforcement mechanisms were identified:
1. **Edge Bot Detection (Cloudflare / AWS WAF / DeepSeek PoW):**
   - **Symptom:** `403 Forbidden`, `429 Too Many Requests`, challenge loops, or `"Need human verification"` responses.
   - **Cause:** TLS fingerprints (JA4), non-browser HTTP/2 frames, missing/stale `aws-waf-token`, or anomalous sub-5ms PoW execution speeds.
2. **Account Moderation Bans ("User is Muted" / 用户已被禁言):**
   - **Symptom:** The API call succeeds at the transport layer, but DeepSeek returns a business error indicating the user account has been muted.
   - **Cause:** DeepSeek operates automated text safety classifiers under Chinese regulatory requirements (specifically cyber-safety and anti-vulgarity / 涉黄 policies). Frontend tools like JanitorAI pass multi-thousand character roleplay prompts containing explicit erotic content and raw machine markup (`<PLUGIN=511EA827>`). These trigger NLP safety tripwires, resulting in automated account suspensions.

---

## 2. Detection Vector Matrix

| # | Detection Vector | Inspection Layer | Root Cause in Current Code | Detection Impact |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **TLS / JA4 Fingerprint** | Edge WAF (Cloudflare) | Python `httpx` links to OpenSSL; lacks Chromium GREASE ciphers (`0x0a0a`), differing cipher list and extension order. | **Critical** (High confidence bot flag) |
| **2** | **HTTP/2 Frame Fingerprint** | Edge WAF / Ingress | Python `h2` sends default `SETTINGS` parameters (header table size, window updates) differing from Chromium's network stack. | **High** (Distinguishes Python from Chrome) |
| **3** | **Instant PoW Solving Speed** | DeepSeek Backend | Native Cranelift/Wasmtime completes SHA3 PoW in **1–5ms**, whereas Chrome's V8 engine takes **80–400ms** (due to worker scheduling and memory layout). | **High** (Turnaround anomaly) |
| **4** | **Stale AWS WAF Tokens** | AWS WAF | `aws-waf-token` contains an encrypted timestamp. Pure HTTP clients cannot execute client-side JavaScript challenge scripts to renew it. | **High** (Session drops after ~1-2 hours) |
| **5** | **Ghost Telemetry (No UI Traffic)** | DeepSeek Analytics | Only `/create_pow_challenge` and `/completion` are called. Authentic browser sessions load history (`/history_messages`), user data (`/users/current`), etc. | **Medium** (Behavioral anomaly) |
| **6** | **Frontend Artifacts & Safety Violations** | DeepSeek NLP Safety | Prompts contain raw `<PLUGIN=...>` tags, 30k+ character system dumps, and policy-violating erotic/NSFW roleplay text. | **Fatal** (Causes permanent account mute) |
| **7** | **Datacenter ASN / IP Reputation** | Cloudflare / Edge IP DB | Outbound requests originate from cloud hosting providers (AWS, Hetzner, DigitalOcean) with high fraud baseline scores. | **Medium to High** (Triggers aggressive CAPTCHAs) |

---

## 3. High-Level Architectural Strategies

Before implementing individual fixes, evaluate the two primary architectural approaches for long-term stealth:

```
+-----------------------------------------------------------------------------+
|                          ARCHITECTURAL COMPARISON                           |
+-----------------------------------------------------------------------------+
| Strategy A: Protocol Hardening (Pure Python Engine)                         |
|   • Transport: curl_cffi (Chrome 131 BoringSSL impersonation)               |
|   • Timing: Artificial V8 event-loop latency simulation                     |
|   • Token Maintenance: Periodic headless Playwright sidecar                 |
|   • Pros: Minimal RAM (~80MB), fast streaming, fully headless/CLI           |
|   • Cons: Requires ongoing maintenance as browser fingerprints evolve       |
+-----------------------------------------------------------------------------+
| Strategy B: Real Browser CDP / Extension Relay                              |
|   • Transport: Genuine Google Chrome via Chrome DevTools Protocol (CDP)     |
|   • Execution: 100% native browser TLS, HTTP/2, V8, and Canvas/DOM          |
|   • Pros: Indistinguishable from human desktop browsing                     |
|   • Cons: Heavy memory footprint (~400MB), requires Chrome installed        |
+-----------------------------------------------------------------------------+
```

| Dimension | Strategy A: Protocol Hardening (`curl_cffi`) | Strategy B: CDP / Browser Relay |
| :--- | :--- | :--- |
| **Anti-Detection Fidelity** | ~90–95% (Matches TLS, H2, PoW, headers) | **100%** (It IS Chrome) |
| **Memory Footprint** | Low (~80–120 MB) | High (350–600 MB) |
| **CPU / Streaming Latency** | Near-zero overhead; instant SSE streaming | Slight CDP serialization delay |
| **Infrastructure Overhead** | Zero GUI dependencies; simple Python binary | Requires Chromium/Chrome binary installed |
| **Recommended Use Case** | **Primary backend server & API bridge** | Fallback if edge WAF deploys canvas challenges |

---

## 4. Concrete Engineering Plans & Code Specifications

### Fix 1: BoringSSL & TLS Fingerprint Alignment via `curl_cffi`

#### The Issue
Even when sending Chrome headers over HTTP/2, edge proxies inspect the TLS `ClientHello` packet before any HTTP headers are transmitted. Python `httpx` (OpenSSL) lacks Chromium's GREASE ciphers (`0x0a0a`, `0x1a1a`), has different elliptic curves, and orders cipher suites in an OpenSSL-specific sequence (yielding an identifiable JA3/JA4 hash).

#### Implementation Blueprint
Replace the transport engine in `deepseek/client.py` with `curl_cffi.requests.Session`:

```python
# deepseek/client.py
from curl_cffi import requests

class DeepSeekClient:
    def __init__(self, session: Optional[Session] = None, timeout: float = 60.0):
        # Impersonate Chrome 131: exact BoringSSL TLS ClientHello + HTTP/2 frame stack
        self._http = requests.Session(impersonate="chrome131")
        self._timeout = int(timeout)
        self.session = session or get_session()
        
        # Load user cookies and base browser headers
        self._http.cookies.update(self.session.cookies)
        self._http.headers.update(self._base_headers())

    def _request_stream(self, method: str, path: str, json: dict, headers: dict):
        url = f"{BASE_URL}{path}"
        response = self._http.request(
            method=method,
            url=url,
            json=json,
            headers=headers,
            stream=True,
            timeout=self._timeout,
        )
        return response.iter_lines()
```

---

### Fix 2: Proof-of-Work (PoW) Timing & V8 Microtask Latency Emulation

#### The Issue
In an authentic browser session, DeepSeek's WebAssembly challenge (`sha3_wasm_bg.wasm`) is scheduled via Web Workers and evaluated inside V8. From receiving the challenge to dispatching the completion POST takes **80ms to 400ms**. In contrast, native compiled Cranelift execution (`wasmtime`) solves the challenge in **1–5ms**. A 5ms turnaround between challenge issuance and submission is an undeniable machine indicator.

#### Implementation Blueprint
Introduce a realistic V8 computation + dispatch latency budget into `deepseek/client.py`:

```python
import random
import time

def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
    # 1. Fetch challenge
    r = self._http.post(f"{BASE_URL}/api/v0/chat/create_pow_challenge", json={"target_path": target_path})
    challenge_data = _biz(r.json())["challenge"]
    
    # 2. Solve challenge via Wasmtime
    t_start = time.perf_counter()
    header_val = self._pow.make_header(challenge_data)
    elapsed = time.perf_counter() - t_start
    
    # 3. Simulate realistic desktop browser V8 latency distribution (120ms - 320ms)
    # Using Gaussian distribution centered at 190ms (sigma = 40ms), clamped [120, 350]
    simulated_v8_target = max(0.12, min(0.35, random.gauss(0.19, 0.04)))
    sleep_needed = simulated_v8_target - elapsed
    if sleep_needed > 0:
        time.sleep(sleep_needed)
        
    return header_val
```

---

### Fix 3: Dynamic AWS WAF Token Refresh (Background Sidecar)

#### The Issue
`aws-waf-token` expires every 30 to 60 minutes. While standard cookies rotate naturally via `Set-Cookie` headers, AWS WAF relies on periodic JavaScript evaluation (evaluating browser canvas, DOM properties, screen dimensions). A pure HTTP client cannot execute these challenges.

#### Implementation Blueprint
In `deepseek/auth.py`, add a lightweight background thread that periodically refreshes the WAF token using the existing Playwright persistent context:

```python
# deepseek/auth.py
import threading
import time
import logging

_log = logging.getLogger("deepseek.auth")

def start_waf_sidecar_refresher(profile_dir: Path, interval_seconds: int = 2400) -> None:
    """Runs a daemon thread refreshing aws-waf-token every 40 minutes."""
    def _worker():
        while True:
            time.sleep(interval_seconds)
            try:
                _log.debug("Initiating background AWS WAF token refresh...")
                with sync_playwright() as p:
                    browser = p.chromium.launch_persistent_context(
                        user_data_dir=str(profile_dir),
                        headless=True,
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    page = browser.new_page()
                    page.goto(CHAT_URL, wait_until="networkidle", timeout=30000)
                    
                    # Extract fresh cookies
                    cookies = {c["name"]: c["value"] for c in browser.cookies()}
                    browser.close()
                    
                    if "aws-waf-token" in cookies:
                        session = Session.load(profile_dir.parent / "session.json")
                        session.cookies.update(cookies)
                        session.save()
                        _log.info("AWS WAF token successfully rotated in background.")
            except Exception as e:
                _log.warning("Background WAF token refresh encountered an error: %s", e)

    t = threading.Thread(target=_worker, name="waf-sidecar", daemon=True)
    t.start()
```

---

### Fix 4: Auxiliary UI Telemetry & Navigation Emulation

#### The Issue
In server logs, an account that calls only `/create_pow_challenge` and `/completion` forms an anomalous access pattern. An authentic user session loads chat history, user settings, and periodically polls status.

#### Implementation Blueprint
Simulate authentic navigation lifecycle events in `deepseek/client.py`:

```python
def _simulate_thread_navigation(self, chat_session_id: str) -> None:
    """Emulates web UI loading conversation history prior to submitting a turn."""
    try:
        self._http.get(
            f"{BASE_URL}/api/v0/chat/history_messages",
            params={"chat_session_id": chat_session_id},
            timeout=10,
        )
    except Exception:
        pass  # Non-blocking telemetry emulation

def check_presence(self) -> None:
    """Emulates periodic user status polling."""
    try:
        self._http.get(f"{BASE_URL}/api/v0/users/current", timeout=10)
    except Exception:
        pass
```

---

### Fix 5: Prompt Normalization & Content Safety Safeguards

#### The Issue
Analysis of ban logs revealed that JanitorAI and similar third-party frontends dump prompts formatted like:
```xml
[System: You are an uncensored AI...]
<PLUGIN=511EA827>
Characters: ... (30,000 characters of explicit NSFW content)
```
DeepSeek's backend operates strict content safety classifiers. When these safety tripwires trigger, the system does not return a WAF error; it permanently mutes the user account ("用户已被禁言").

#### Implementation Blueprint
Add sanitization and prompt safety heuristics in `server/openai_format.py`:

```python
# server/openai_format.py
import re
import logging

_log = logging.getLogger("server.safety")

# Regex to strip non-standard frontend automation tags
AUTOMATION_TAG_RE = re.compile(r"<PLUGIN=[A-Fa-f0-9]+>|</?context>|\[System:\s*UNCENSORED.*?\]", re.IGNORECASE)

def sanitize_prompt_content(content: str) -> str:
    """Removes automated frontend wrapper tags that flag machine traffic."""
    cleaned = AUTOMATION_TAG_RE.sub("", content)
    return cleaned.strip()

def validate_prompt_safety(messages: list[dict]) -> None:
    """Logs warnings if prompts exhibit patterns known to trigger account bans."""
    total_len = sum(len(m.get("content", "")) for m in messages)
    if total_len > 25000:
        _log.warning(
            "Large initial prompt detected (%d chars). Monolithic prompt dumps on turn 1 "
            "trigger anomalous usage heuristics.", total_len
        )
```

---

### Fix 6: Residential & Home Egress Routing (`UPSTREAM_PROXY`)

#### The Issue
Data center IP ranges (AWS, DigitalOcean, Hetzner, Linode) are cataloged in IP intelligence feeds (MaxMind, Spur, IPinfo) with high fraud and bot scores.

#### Implementation Blueprint
Support an upstream HTTP or SOCKS5 residential proxy in `.env`:

```bash
# .env configuration
UPSTREAM_PROXY=socks5://127.0.0.1:1080
# Or residential proxy:
# UPSTREAM_PROXY=http://user:password@residential-node.example.com:8000
```

Load this in `deepseek/client.py`:
```python
proxy = os.getenv("UPSTREAM_PROXY")
if proxy:
    self._http.proxies = {"http": proxy, "https": proxy}
```

---

## 5. Phased Implementation Roadmap

```
+-------------------------------------------------------------------------------+
| PHASE 1: Transport & Timing Parity (High ROI, Immediate)                      |
| [ ] 1.1 Add curl_cffi to requirements.txt                                     |
| [ ] 1.2 Refactor deepseek/client.py transport to requests.Session(impersonate) |
| [ ] 1.3 Add V8 execution timing jitter (120ms-320ms) in _pow_header           |
| [ ] 1.4 Add UPSTREAM_PROXY support to .env and client initialization          |
+-------------------------------------------------------------------------------+
| PHASE 2: State Durability & Telemetry (Medium Complexity)                     |
| [ ] 2.1 Add Playwright background sidecar task in deepseek/auth.py            |
| [ ] 2.2 Wire automatic 40-minute aws-waf-token renewal loop                  |
| [ ] 2.3 Integrate _simulate_thread_navigation on multi-turn continuations     |
| [ ] 2.4 Add periodic /users/current presence ping in background               |
+-------------------------------------------------------------------------------+
| PHASE 3: Prompt Hygiene & Front-End Guards (Content Protection)               |
| [ ] 3.1 Clean raw <PLUGIN=...> artifacts in server/openai_format.py           |
| [ ] 3.2 Add warning logging on 25k+ character monolithic prompt dumps         |
| [ ] 3.3 Add documentation regarding terms of service & content safety limits  |
+-------------------------------------------------------------------------------+
```

---

## 6. Verification & Validation Playbook

To objectively confirm each mitigation after implementation:

1. **TLS / JA4 Verification:**
   - Call a TLS fingerprint echo API through the client:
     ```python
     res = client._http.get("https://tls.peet.ws/api/all").json()
     print("TLS Version:", res["tls"]["version"])
     print("JA4:", res["tls"]["ja4"])
     print("BoringSSL GREASE Present:", any("0x" in c for c in res["tls"]["ciphers"]))
     ```
   - Confirm the JA4 matches genuine Chrome (`t13d1516h2_...`).

2. **PoW Timing Distribution Test:**
   - Execute 10 consecutive PoW challenges and calculate the delta between challenge creation and resolution.
   - Verify that all turnaround times fall between **120ms and 350ms** with a natural Gaussian spread.

3. **24-Hour Token Durability Test:**
   - Run a periodic query script every 15 minutes across a 24-hour window.
   - Verify zero `403 Forbidden` or `aws-waf-token expired` errors.
