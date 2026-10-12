"""The Chrome extension, end to end, against a mock DeepSeek.

Loads extension/ into Playwright's Chromium, runs the real server with Gmail
faked, and serves a stand-in for chat.deepseek.com whose sign-up page has the
same fields and buttons. The first "Send code" answers with the "Current
device environment error" toast and the second with "Registration failed.
Please try again later (B11)", so the refresh-and-retry path runs too. Checks that the
form was filled with the job's credentials, the code was entered, and the
captured token and cookies landed in the session file.

Needs a browser and takes about 15 s per case, so it runs
only when asked:

    RUN_EXTENSION_E2E=1 venv/bin/python -m unittest tests.test_extension_e2e
"""

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXTENSION_DIR = ROOT / "extension"

MOCK_STATE = {"sends": [], "signups": [], "toasts": []}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SIGN_IN = """<!doctype html><title>DeepSeek</title>
<h1>Sign in</h1>
<div role="button" id="to-signup"><span class="ds-button__content">Sign up</span></div>
<script>document.getElementById('to-signup').addEventListener('click', () => location.href = '/sign_up');</script>
"""

# Values are tracked through input events, the way a React form would see
# them, so a value poked straight into .value would not count.
SIGN_UP = """<!doctype html><title>DeepSeek sign up</title>
<h1>Sign up</h1>
<input placeholder="Email address"><input placeholder="Password" type="password">
<input placeholder="Confirm password" type="password">
<div role="button" id="send"><span class="ds-button__content">Send code</span></div>
<div id="toast"></div>
<div id="step2" style="display:none">
  <input placeholder="Code">
  <div role="button" id="signup"><span class="ds-button__content">Sign up</span></div>
</div>
<script>
const seen = {};
for (const el of document.querySelectorAll('input')) {
  el.addEventListener('input', () => { seen[el.placeholder] = el.value; });
}
document.getElementById('send').addEventListener('click', async () => {
  const r = await fetch('/mock/send', {method: 'POST', body: JSON.stringify(seen)});
  const d = await r.json();
  if (d.toast) {
    document.getElementById('toast').textContent = d.toast;
  } else {
    document.getElementById('step2').style.display = 'block';
  }
});
document.getElementById('signup').addEventListener('click', async () => {
  const r = await fetch('/mock/signup', {method: 'POST', body: JSON.stringify({code: seen['Code']})});
  if ((await r.json()).ok) {
    localStorage.setItem('userToken', JSON.stringify({value: 'tok-e2e', __version: '0'}));
    location.href = '/';
  }
});
</script>
"""

CHAT = "<!doctype html><title>DeepSeek</title><h1>chat</h1>"


class MockDeepSeek(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _page(self, html, headers=()):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/sign_in":
            return self._page(SIGN_IN)
        if path == "/sign_up":
            return self._page(SIGN_UP)
        if path == "/":
            return self._page(CHAT)
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        if self.path == "/mock/send":
            MOCK_STATE["sends"].append(body)
            toast = MOCK_STATE["toasts"].pop(0) if MOCK_STATE["toasts"] else None
            if not toast:
                # "The verification email arrives."
                from tests.test_register import FakeGmail
                FakeGmail.codes[body.get("Email address", "")] = "246810"
            return self._json({"toast": toast})
        if self.path == "/mock/signup":
            MOCK_STATE["signups"].append(body)
            return self._json({"ok": True}, headers=[("Set-Cookie", "ds_session=cookie-e2e; Path=/")])
        self.send_response(404)
        self.end_headers()

    def _json(self, data, headers=()):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(os.getenv("RUN_EXTENSION_E2E") == "1",
                     "set RUN_EXTENSION_E2E=1 to drive the extension in Chromium")
class ExtensionEndToEnd(unittest.TestCase):
    def setUp(self):
        from tests.test_register import FakeGmail
        from server import register
        FakeGmail.registered, FakeGmail.highest, FakeGmail.codes = set(), None, {}
        MOCK_STATE.update(sends=[], signups=[], toasts=[
            "Current device environment error. Please try a different environment.",
            "Registration failed. Please try again later (B11)",
        ])

        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.env_file = tmp / ".env"
        self.env_file.write_text("DEEPSEEK_EMAIL=user+3@gmail.com\nDEEPSEEK_PASSWORD=pw123\n"
                                 "GMAIL_APP_PASSWORD=app-pw\n")
        self.session_file = tmp / "session.json"
        self.patches = [
            mock.patch.object(register, "Gmail", FakeGmail),
            mock.patch.object(register, "ACCOUNT_ENV", self.env_file),
            mock.patch.object(register, "SESSION_FILE", self.session_file),
            mock.patch.object(register, "broker", register.Broker()),
            mock.patch.object(register, "session_installed", None),
        ]
        for p in self.patches:
            p.start()
        for key in ("DEEPSEEK_EMAIL", "DEEPSEEK_PASSWORD", "GMAIL_EMAIL", "GMAIL_APP_PASSWORD"):
            os.environ.pop(key, None)

        # The mock DeepSeek.
        self.mock_port = free_port()
        self.mock = ThreadingHTTPServer(("127.0.0.1", self.mock_port), MockDeepSeek)
        threading.Thread(target=self.mock.serve_forever, daemon=True).start()

        # The real server.
        import uvicorn
        from server.api import app
        self.api_port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.api_port,
                                                    log_level="warning"))
        threading.Thread(target=self.server.run, daemon=True).start()
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.api_port}/healthz", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

    def tearDown(self):
        self.server.should_exit = True
        self.mock.shutdown()
        self.mock.server_close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_registers_through_the_extension(self):
        self._register(trusted=False)

    def test_registers_with_trusted_input(self):
        """The chrome.debugger path: real key and mouse events, as register.py
        sends them. The "debugger" permission is optional and only granted on
        a click, so the test loads a copy of the extension that requires it."""
        self._register(trusted=True)

    def _register(self, trusted: bool):
        import shutil
        import urllib.request
        from playwright.sync_api import sync_playwright

        mock_base = f"http://127.0.0.1:{self.mock_port}"
        api_base = f"http://127.0.0.1:{self.api_port}"
        with sync_playwright() as pw, tempfile.TemporaryDirectory() as profile, \
                tempfile.TemporaryDirectory() as ext_copy:
            ext_dir = EXTENSION_DIR
            if trusted:
                ext_dir = Path(ext_copy) / "extension"
                shutil.copytree(EXTENSION_DIR, ext_dir)
                manifest = json.loads((ext_dir / "manifest.json").read_text())
                manifest["permissions"].append("debugger")
                manifest["optional_permissions"] = []
                (ext_dir / "manifest.json").write_text(json.dumps(manifest))
            context = pw.chromium.launch_persistent_context(
                profile, channel="chromium", headless=True,
                args=[f"--disable-extensions-except={ext_dir}",
                      f"--load-extension={ext_dir}"])
            try:
                worker = context.service_workers[0] if context.service_workers \
                    else context.wait_for_event("serviceworker", timeout=15000)
                ext_id = worker.url.split("/")[2]

                # Point the extension at this test's servers, then wake it.
                popup = context.new_page()
                popup.goto(f"chrome-extension://{ext_id}/popup.html")
                popup.evaluate("""(s) => chrome.storage.local.set(s)""", {
                    "server": api_base, "signinUrl": f"{mock_base}/sign_in",
                    "signupUrl": f"{mock_base}/sign_up", "chatUrl": f"{mock_base}/",
                    "trustedInput": trusted,
                })
                popup.reload()
                popup.wait_for_selector("text=Server reachable", timeout=10000)
                popup.wait_for_selector("text=Sign-up credentials: configured", timeout=10000)

                req = urllib.request.Request(f"{api_base}/register", method="POST",
                                             data=b'{"reason":"e2e"}',
                                             headers={"Content-Type": "application/json"})
                job = json.loads(urllib.request.urlopen(req, timeout=10).read())
                popup.evaluate("() => chrome.runtime.sendMessage({scope: 'deepseek-register', cmd: 'poll'})")

                deadline = time.time() + 180
                while time.time() < deadline:
                    status = json.loads(urllib.request.urlopen(
                        f"{api_base}/register/jobs/{job['id']}", timeout=10).read())
                    if status["state"] in ("done", "failed"):
                        break
                    time.sleep(1)
                # The popup follows the job.
                popup.wait_for_selector(f"text=Job {job['id']}: {status['state']}", timeout=10000)
            finally:
                context.close()

        log = "\n".join(f"{e['stage']}: {e['message']}" for e in status["log"])
        self.assertEqual(status["state"], "done", log)
        self.assertIn("trusted input" if trusted else "synthetic input", log)

        # Three sends: device error, then B11, then it went through.
        self.assertEqual(len(MOCK_STATE["sends"]), 3, MOCK_STATE)
        for send in MOCK_STATE["sends"]:
            self.assertEqual(send["Email address"], "user+3@gmail.com")
            self.assertEqual(send["Password"], "pw123")
            self.assertEqual(send["Confirm password"], "pw123")
        self.assertEqual(MOCK_STATE["signups"], [{"code": "246810"}])
        self.assertIn("Current device environment error", log)
        self.assertIn("Registration failed. Please try again later", log)
        # Retries stayed on the sign-up page: the sign-in page was loaded once.
        self.assertEqual(log.count("opening /sign_up directly"), 0)

        saved = json.loads(self.session_file.read_text())
        self.assertEqual(saved["token"], "tok-e2e")
        self.assertEqual(saved["cookies"], {"ds_session": "cookie-e2e"})
        self.assertIn("Chrome/", saved["user_agent"])
        self.assertNotIn("HeadlessChrome", saved["user_agent"])


if __name__ == "__main__":
    unittest.main()
