import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from deepseek.auth import Session, clean_user_agent
from deepseek.client import _client_hints, chrome_major, impersonation_target, sec_ch_ua

HEADLESS_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
               "HeadlessChrome/148.0.0.0 Safari/537.36")


class SecChUa(unittest.TestCase):
    """Headers real Google Chrome sends; 148 was captured from this machine's
    Chrome 148.0.7778.96 on 2026-09-25, the rest are well-known releases."""

    KNOWN = {
        120: '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        124: '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        131: '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        148: '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    }

    def test_matches_real_chrome(self):
        for major, header in self.KNOWN.items():
            self.assertEqual(sec_ch_ua(major), header, major)

    def test_client_hints_use_the_user_agents_version(self):
        h = _client_hints(clean_user_agent(HEADLESS_UA))
        self.assertEqual(h["sec-ch-ua"], self.KNOWN[148])
        self.assertEqual(h["sec-ch-ua-platform"], '"Linux"')


class Impersonation(unittest.TestCase):
    def test_newest_fingerprint_not_newer_than_the_browser(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CURL_IMPERSONATE", None)
            self.assertEqual(impersonation_target("Chrome/131.0.0.0"), "chrome131")
            self.assertEqual(impersonation_target("Chrome/148.0.0.0"), "chrome146")
            self.assertEqual(impersonation_target("Chrome/154.0.0.0"), "chrome150")
            self.assertEqual(impersonation_target("Chrome/140.0.0.0"), "chrome136")

    def test_override(self):
        with mock.patch.dict(os.environ, {"CURL_IMPERSONATE": "chrome124"}):
            self.assertEqual(impersonation_target("Chrome/148.0.0.0"), "chrome124")

    def test_no_version_falls_back_to_the_default_user_agents(self):
        os.environ.pop("CURL_IMPERSONATE", None)
        self.assertEqual(chrome_major(""), 131)
        self.assertEqual(impersonation_target(""), "chrome131")


class HeadlessUserAgent(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(clean_user_agent(HEADLESS_UA), HEADLESS_UA.replace("HeadlessChrome", "Chrome"))
        self.assertEqual(clean_user_agent(""), "")
        self.assertEqual(clean_user_agent(None), "")

    def test_session_is_cleaned_when_captured_and_when_loaded(self):
        s = Session(token="t", cookies={}, user_agent=HEADLESS_UA, captured_at=0.0)
        self.assertNotIn("Headless", s.user_agent)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "session.json"
            path.write_text(json.dumps({"token": "t", "cookies": {}, "user_agent": HEADLESS_UA,
                                        "captured_at": 0.0}))
            loaded = Session.load(path)
        self.assertNotIn("Headless", loaded.user_agent)
        self.assertIn("Chrome/148.0.0.0", loaded.user_agent)


if __name__ == "__main__":
    unittest.main()
