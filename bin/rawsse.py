#!/usr/bin/env python
"""Send one prompt to DeepSeek with the saved session and print the raw SSE
frames with timestamps. Diagnostic for "the server hangs": run from the
repo root with venv/bin/python bin/rawsse.py [prompt] [model_type]."""
import sys, time, json
sys.path.insert(0, ".")
from deepseek.client import DeepSeekClient, COMPLETION_PATH
prompt = sys.argv[1] if len(sys.argv) > 1 else "Reply with the single word: pong"
model = sys.argv[2] if len(sys.argv) > 2 else None
c = DeepSeekClient(allow_interactive=False)
print("client ok, curl:", getattr(c, "_use_curl", False), flush=True)
sid = c.create_chat_session()
print("session", sid, flush=True)
body = {"chat_session_id": sid, "parent_message_id": None, "prompt": prompt,
        "ref_file_ids": [], "thinking_enabled": False, "search_enabled": False,
        "action": None, "preempt": False}
if model: body["model_type"] = model
t0 = time.time()
resp = c._http.post(COMPLETION_PATH, json=body, headers={"x-ds-pow-response": c._pow_header()}, stream=True, timeout=120)
print(f"HTTP {resp.status_code} after {time.time()-t0:.1f}s; headers: {dict(resp.headers)}", flush=True)
n = 0
for line in resp.iter_lines():
    if isinstance(line, bytes): line = line.decode("utf-8", "replace")
    if not line: continue
    n += 1
    if n <= 40 or n % 50 == 0:
        print(f"[{time.time()-t0:6.1f}s] {line[:300]}", flush=True)
print(f"stream closed after {time.time()-t0:.1f}s, {n} lines", flush=True)
