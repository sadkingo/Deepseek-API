"""server/register.py: the broker between callers and the Chrome extension.

Gmail is replaced by a fake; the routes are exercised through the test client
the way the extension and `python -m server.register` use them.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from server import register
from server.api import app


class FakeGmail:
    """What the inbox answers, scriptable per test."""

    registered = set()
    highest = None
    codes = {}          # address -> code that "arrives"
    latest = 100
    calls = []

    def __init__(self, user, app_password):
        self.user, self.app_password = user, app_password

    def latest_uid(self):
        FakeGmail.calls.append("latest_uid")
        return FakeGmail.latest

    def is_registered(self, addr):
        return addr in FakeGmail.registered

    def highest_registered_num(self):
        return FakeGmail.highest

    resolve_next_email = register.Gmail.resolve_next_email

    def wait_for_code(self, after_uid, for_address, deadline, stop, on_status=lambda s: None):
        FakeGmail.calls.append(("wait", after_uid, for_address))
        on_status("connected")
        while time.time() < deadline and not stop.is_set():
            if for_address in FakeGmail.codes:
                return FakeGmail.codes[for_address]
            stop.wait(0.05)
        return None


class RegisterRoutes(unittest.TestCase):
    def setUp(self):
        FakeGmail.registered = set()
        FakeGmail.highest = None
        FakeGmail.codes = {}
        FakeGmail.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.env_file = tmp / ".env"
        self.env_file.write_text(
            "DEEPSEEK_EMAIL=user+3@gmail.com\nDEEPSEEK_PASSWORD=pw123\n"
            "GMAIL_APP_PASSWORD=app-pw\n", encoding="utf-8")
        self.session_file = tmp / "session" / "session.json"
        self.patches = [
            mock.patch.object(register, "Gmail", FakeGmail),
            mock.patch.object(register, "ACCOUNT_ENV", self.env_file),
            mock.patch.object(register, "SESSION_FILE", self.session_file),
            mock.patch.object(register, "broker", register.Broker()),
            mock.patch.object(register, "CODE_WAIT_SECONDS", 2.0),
            # Do not swap the shared server client for one built on fake data.
            mock.patch.object(register, "session_installed", None),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in self.patches:
            p.start()
        for key in ("DEEPSEEK_EMAIL", "DEEPSEEK_PASSWORD", "GMAIL_EMAIL", "GMAIL_APP_PASSWORD"):
            os.environ.pop(key, None)
        self.client = TestClient(app)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    # -- the whole exchange, as the extension does it ----------------------
    def test_full_exchange_saves_and_installs_the_session(self):
        installed = []
        with mock.patch.object(register, "session_installed", installed.append):
            job = self.client.post("/register", json={"reason": "test"}).json()
            self.assertEqual(job["state"], "pending")
            self.assertEqual(job["email"], "us***@gmail.com")  # masked for callers
            self.assertNotIn("password", job)

            # The extension picks it up and gets the real credentials.
            picked = self.client.get("/register/jobs/next?wait=1").json()
            self.assertEqual(picked["id"], job["id"])
            self.assertEqual(picked["email"], "user+3@gmail.com")
            self.assertEqual(picked["password"], "pw123")
            self.assertEqual(picked["state"], "running")
            # Nothing else is handed out while it runs.
            self.assertEqual(self.client.get("/register/jobs/next?wait=0").status_code, 204)

            # "send_code" pins the inbox baseline.
            FakeGmail.latest = 41
            r = self.client.post(f"/register/jobs/{job['id']}/progress",
                                 json={"stage": "send_code", "message": "clicking"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(register.broker.get(job["id"]).baseline_uid, 41)

            # The code is waited for and arrives.
            r = self.client.get(f"/register/jobs/{job['id']}/code?wait=0.2").json()
            self.assertEqual(r, {"pending": True})
            FakeGmail.codes["user+3@gmail.com"] = "123456"
            r = self.client.get(f"/register/jobs/{job['id']}/code?wait=2").json()
            self.assertEqual(r, {"code": "123456"})
            self.assertIn(("wait", 41, "user+3@gmail.com"), FakeGmail.calls)

            r = self.client.post(f"/register/jobs/{job['id']}/result", json={
                "token": "tok-abc", "cookies": {"a": "1", "b": "2"},
                "user_agent": "Mozilla/5.0 HeadlessChrome/154.0 Safari/537.36",
            })
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["state"], "done")

        saved = json.loads(self.session_file.read_text())
        self.assertEqual(saved["token"], "tok-abc")
        self.assertEqual(saved["cookies"], {"a": "1", "b": "2"})
        self.assertNotIn("HeadlessChrome", saved["user_agent"])
        self.assertEqual(len(installed), 1)
        self.assertEqual(installed[0].token, "tok-abc")
        # The account the server signs back into is the one just made.
        self.assertIn("DEEPSEEK_EMAIL=user+3@gmail.com", self.env_file.read_text())

        status = self.client.get(f"/register/jobs/{job['id']}").json()
        self.assertEqual(status["state"], "done")
        self.assertTrue(any("session saved" in e["message"] for e in status["log"]))

    def test_each_send_code_gets_its_own_code(self):
        """A retry clicks "Send code" again: the earlier code (or the watcher
        still waiting for it) must not be handed to the new attempt."""
        job = self.client.post("/register").json()
        self.client.get("/register/jobs/next?wait=1")
        FakeGmail.latest = 10
        self.client.post(f"/register/jobs/{job['id']}/progress", json={"stage": "send_code"})
        FakeGmail.codes["user+3@gmail.com"] = "111111"
        r = self.client.get(f"/register/jobs/{job['id']}/code?wait=2").json()
        self.assertEqual(r, {"code": "111111"})

        # The page showed "Registration failed (B11)": refresh, fill, send again.
        FakeGmail.latest = 20
        del FakeGmail.codes["user+3@gmail.com"]
        self.client.post(f"/register/jobs/{job['id']}/progress", json={"stage": "send_code"})
        self.assertEqual(register.broker.get(job["id"]).baseline_uid, 20)
        r = self.client.get(f"/register/jobs/{job['id']}/code?wait=0.2").json()
        self.assertEqual(r, {"pending": True}, "the old code must not be reused")
        FakeGmail.codes["user+3@gmail.com"] = "222222"
        r = self.client.get(f"/register/jobs/{job['id']}/code?wait=2").json()
        self.assertEqual(r, {"code": "222222"})
        self.assertIn(("wait", 20, "user+3@gmail.com"), FakeGmail.calls)

    def test_next_free_alias_is_used_and_recorded_on_success(self):
        FakeGmail.registered = {"user+3@gmail.com"}
        FakeGmail.highest = 7
        job = self.client.get("/register/jobs/next?wait=0", headers={}).status_code
        self.assertEqual(job, 204)
        job = self.client.post("/register").json()
        picked = self.client.get("/register/jobs/next?wait=1").json()
        self.assertEqual(picked["email"], "user+8@gmail.com")
        self.assertIn("DEEPSEEK_EMAIL=user+3@gmail.com", self.env_file.read_text())
        self.client.post(f"/register/jobs/{job['id']}/result",
                         json={"token": "t", "cookies": {}, "user_agent": "ua"})
        self.assertIn("DEEPSEEK_EMAIL=user+8@gmail.com", self.env_file.read_text())

    def test_failure_and_requeue(self):
        job = self.client.post("/register").json()
        again = self.client.post("/register").json()
        self.assertEqual(again["id"], job["id"], "one job at a time")
        self.client.get("/register/jobs/next?wait=1")
        r = self.client.post(f"/register/jobs/{job['id']}/fail", json={"error": "device error x5"})
        self.assertEqual(r.json()["state"], "failed")
        self.assertEqual(r.json()["error"], "device error x5")
        fresh = self.client.post("/register").json()
        self.assertNotEqual(fresh["id"], job["id"])
        self.assertFalse(self.session_file.exists())

    def test_result_without_anything_captured_fails_the_job(self):
        job = self.client.post("/register").json()
        self.client.get("/register/jobs/next?wait=1")
        r = self.client.post(f"/register/jobs/{job['id']}/result",
                             json={"token": "", "cookies": {}, "user_agent": "x"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.get(f"/register/jobs/{job['id']}").json()["state"], "failed")

    def test_code_timeout_is_reported(self):
        with mock.patch.object(register, "CODE_WAIT_SECONDS", 0.3):
            job = self.client.post("/register").json()
            self.client.get("/register/jobs/next?wait=1")
            r = self.client.get(f"/register/jobs/{job['id']}/code?wait=2").json()
        self.assertIn("no verification code within", r["error"])

    def test_stale_job_expires(self):
        job = self.client.post("/register").json()
        self.client.get("/register/jobs/next?wait=1")
        with mock.patch.object(register, "JOB_STALE_SECONDS", 0.0):
            status = self.client.get(f"/register/jobs/{job['id']}").json()
        self.assertEqual(status["state"], "failed")
        self.assertIn("no word from the extension", status["error"])

    def test_wait_blocks_until_done(self):
        def finish_soon():
            time.sleep(0.3)
            picked = self.client.get("/register/jobs/next?wait=1").json()
            self.client.post(f"/register/jobs/{picked['id']}/result",
                             json={"token": "t", "cookies": {}, "user_agent": "ua"})
        threading.Thread(target=finish_soon, daemon=True).start()
        job = self.client.post("/register", json={"wait": 5}).json()
        self.assertEqual(job["state"], "done")

    def test_missing_credentials_is_a_503(self):
        self.env_file.write_text("DEEPSEEK_EMAIL=x@y.z\n", encoding="utf-8")
        r = self.client.post("/register")
        self.assertEqual(r.status_code, 503)
        self.assertIn("DEEPSEEK_PASSWORD", r.json()["error"])
        self.assertIn("GMAIL_APP_PASSWORD", r.json()["error"])
        self.assertEqual(self.client.get("/register").json()["credentials_missing"],
                         ["DEEPSEEK_PASSWORD", "GMAIL_APP_PASSWORD"])

    # -- who may call --------------------------------------------------------
    def test_only_direct_local_callers(self):
        for headers, why in [
            ({"Origin": "https://some.site"}, "web pages"),
            ({"X-Forwarded-For": "10.0.0.5"}, "proxy or tunnel"),
            ({"CF-Connecting-IP": "1.1.1.1"}, "proxy or tunnel"),
        ]:
            r = self.client.get("/register", headers=headers)
            self.assertEqual(r.status_code, 403, headers)
            self.assertIn(why, r.json()["error"])
        r = self.client.get("/register", headers={"Origin": "chrome-extension://abcdef"})
        self.assertEqual(r.status_code, 200)

    def test_shared_token_when_configured(self):
        with mock.patch.object(register, "REGISTER_TOKEN", "s3cret"):
            self.assertEqual(self.client.get("/register").status_code, 401)
            r = self.client.get("/register", headers={"X-Register-Token": "s3cret"})
            self.assertEqual(r.status_code, 200)


class CommandLine(unittest.TestCase):
    """`python -m server.register` with no API server running: it hosts the
    routes itself for the one job. A fake extension does the exchange over
    HTTP, as the real one would."""

    def test_hosts_the_broker_when_the_server_is_down(self):
        import io
        import socket
        import urllib.request
        from contextlib import redirect_stdout

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env_file = Path(tmp.name) / ".env"
        env_file.write_text("DEEPSEEK_EMAIL=user+3@gmail.com\nDEEPSEEK_PASSWORD=pw123\n"
                            "GMAIL_APP_PASSWORD=app-pw\n")
        session_file = Path(tmp.name) / "session.json"
        FakeGmail.registered, FakeGmail.highest, FakeGmail.codes = set(), None, {}

        def call(method, path, body=None):
            req = urllib.request.Request(base + path, method=method,
                                         data=json.dumps(body).encode() if body else None,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read()
            return json.loads(raw) if raw else None

        def fake_extension():
            for _ in range(100):  # the broker comes up a moment after main() starts
                try:
                    job = call("GET", "/register/jobs/next?wait=1")
                except Exception:
                    time.sleep(0.1)
                    continue
                if job:
                    break
            call("POST", f"/register/jobs/{job['id']}/progress",
                 {"stage": "send_code", "message": "clicking"})
            FakeGmail.codes[job["email"]] = "111222"
            code = call("GET", f"/register/jobs/{job['id']}/code?wait=5")
            self.assertEqual(code, {"code": "111222"})
            call("POST", f"/register/jobs/{job['id']}/result",
                 {"token": "tok-cli", "cookies": {"c": "1"}, "user_agent": "ua"})

        t = threading.Thread(target=fake_extension, daemon=True)
        with mock.patch.object(register, "Gmail", FakeGmail), \
                mock.patch.object(register, "ACCOUNT_ENV", env_file), \
                mock.patch.object(register, "SESSION_FILE", session_file), \
                mock.patch.object(register, "broker", register.Broker()), \
                mock.patch.object(register, "session_installed", None), \
                mock.patch.dict(os.environ, {}, clear=False):
            for key in ("DEEPSEEK_EMAIL", "DEEPSEEK_PASSWORD", "GMAIL_EMAIL", "GMAIL_APP_PASSWORD"):
                os.environ.pop(key, None)
            t.start()
            out = io.StringIO()
            with redirect_stdout(out):
                code = register.main(["--server", base, "--timeout", "30"])
            t.join(5)

        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("not running", out.getvalue())
        self.assertIn("session is saved for the API server", out.getvalue())
        self.assertEqual(json.loads(session_file.read_text())["token"], "tok-cli")
        # The temporary broker is gone again.
        time.sleep(0.5)
        with self.assertRaises(Exception):
            call("GET", "/register")


class MutedRecovery(unittest.TestCase):
    """server/api.py on "user is muted": queue a job, get Chrome up, switch to
    the new session in place; fall back to the exit-and-restart path when the
    extension route fails."""

    def setUp(self):
        from server import api
        self.api = api
        FakeGmail.registered, FakeGmail.highest, FakeGmail.codes = set(), None, {}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env_file = Path(self.tmp.name) / ".env"
        env_file.write_text("DEEPSEEK_EMAIL=user+3@gmail.com\nDEEPSEEK_PASSWORD=pw123\n"
                            "GMAIL_APP_PASSWORD=app-pw\n")
        self.launched = []
        self.restarted = []
        self.installed = []
        self.patches = [
            mock.patch.object(register, "Gmail", FakeGmail),
            mock.patch.object(register, "ACCOUNT_ENV", env_file),
            mock.patch.object(register, "SESSION_FILE", Path(self.tmp.name) / "session.json"),
            mock.patch.object(register, "broker", register.Broker()),
            mock.patch.object(register, "launch_chrome", lambda: self.launched.append(1) or "chrome"),
            mock.patch.object(register, "session_installed", self.installed.append),
            mock.patch.object(register, "CODE_WAIT_SECONDS", 1.0),
            mock.patch.object(api, "MUTED_RECOVERY", "extension"),
            mock.patch.object(api, "_recover_last_at", 0.0),
            mock.patch.object(api, "_recover_in_progress", False),
            mock.patch.object(api, "_recover_by_restart", lambda: self.restarted.append(1)),
            # Keep the simulated mutes out of the real logs/requests.log.
            mock.patch.object(api.debuglog, "log_error", lambda *a, **k: None),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        for key in ("DEEPSEEK_EMAIL", "DEEPSEEK_PASSWORD", "GMAIL_EMAIL", "GMAIL_APP_PASSWORD"):
            os.environ.pop(key, None)

    def _wait(self, cond, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return True
            time.sleep(0.05)
        return cond()

    def test_mute_queues_a_job_and_switches_when_the_session_arrives(self):
        with mock.patch.object(register, "ensure_extension", lambda b, **kw: True):
            self.api._maybe_recover_muted("Upstream error: user is muted")
            self.api._maybe_recover_muted("user is muted")  # a second request: same job
            broker = register.broker
            self.assertTrue(self._wait(lambda: broker.current is not None))
            job = broker.next_pending(1)
            self.assertEqual(job.reason, "account muted")
            self.assertEqual(len(broker.jobs), 1)
            broker.finish(job, {"token": "tok-new", "cookies": {"x": "1"}, "user_agent": "ua"})
            self.assertTrue(self._wait(lambda: not self.api._recover_in_progress))
        self.assertEqual([s.token for s in self.installed], ["tok-new"])
        self.assertEqual(self.restarted, [])

    def test_chrome_is_started_when_the_extension_is_silent(self):
        broker = register.broker
        said = []
        self.assertFalse(register.ensure_extension(broker, grace=0.2, say=said.append))
        self.assertEqual(self.launched, [1])
        self.assertTrue(any("starting Chrome" in s for s in said), said)
        # ...but not when it has just polled.
        broker.next_pending(0)
        self.assertTrue(register.ensure_extension(broker, grace=0.2))
        self.assertEqual(self.launched, [1])

    def test_failure_falls_back_to_restart(self):
        with mock.patch.object(register, "ensure_extension", lambda b, **kw: True):
            self.api._maybe_recover_muted("user is muted")
            broker = register.broker
            self.assertTrue(self._wait(lambda: broker.current is not None))
            broker.fail(broker.current, "device error x5")
            self.assertTrue(self._wait(lambda: self.restarted == [1]))
        self.assertEqual(self.installed, [])

    def test_restart_mode_skips_the_extension(self):
        with mock.patch.object(self.api, "MUTED_RECOVERY", "restart"):
            self.api._maybe_recover_muted("user is muted")
        self.assertEqual(self.restarted, [1])
        self.assertIsNone(register.broker.current)

    def test_not_twice_within_ten_minutes(self):
        with mock.patch.object(register, "ensure_extension", lambda b, **kw: True):
            self.api._maybe_recover_muted("user is muted")
            broker = register.broker
            self.assertTrue(self._wait(lambda: broker.current is not None))
            broker.fail(broker.current, "x")
            self.assertTrue(self._wait(lambda: not self.api._recover_in_progress))
            self.api._maybe_recover_muted("user is muted")
            time.sleep(0.2)
        self.assertEqual(len(broker.jobs), 1)


class Helpers(unittest.TestCase):
    def test_parse_email(self):
        self.assertEqual(register.parse_email("user+41@gmail.com"), ("user", 41))
        self.assertEqual(register.parse_email("user@gmail.com"), ("user", None))
        self.assertEqual(register.parse_email("user+tag@gmail.com"), ("user", None))

    def test_extract_code_ignores_markup_and_css(self):
        html = ("<style>.x{width:123456px}</style><script>var n=654321</script>"
                "<p>Your code is <b>987654</b></p>")
        self.assertEqual(register.extract_code(html), "987654")
        self.assertIsNone(register.extract_code("<p>no code here 12345</p>"))

    def test_update_env_email_adds_or_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text("A=1\nDEEPSEEK_EMAIL=old@x.y\nB=2\n")
            register.update_env_email("new@x.y", path)
            self.assertEqual(path.read_text(), "A=1\nDEEPSEEK_EMAIL=new@x.y\nB=2\n")
            path.write_text("A=1\n")
            register.update_env_email("new@x.y", path)
            self.assertEqual(path.read_text(), "A=1\nDEEPSEEK_EMAIL=new@x.y\n")


if __name__ == "__main__":
    unittest.main()
