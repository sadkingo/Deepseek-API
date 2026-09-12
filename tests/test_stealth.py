import os
import time
import unittest
from unittest.mock import MagicMock, patch

from deepseek.auth import Session, start_waf_sidecar
from deepseek.client import DeepSeekClient
from server.schemas import ChatMessage
from server.openai_format import messages_to_prompt


class TestStealthFeatures(unittest.TestCase):
    def setUp(self):
        self.session = Session(
            token="test-token-123",
            cookies={"test_cookie": "test_value"},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            captured_at=time.time(),
        )

    def test_curl_cffi_client_initialized(self):
        client = DeepSeekClient(session=self.session)
        self.assertTrue(client._use_curl)
        self.assertTrue(hasattr(client._http, "impersonate"))
        self.assertEqual(client._http.impersonate, "chrome131")

    def test_pow_latency_simulation(self):
        client = DeepSeekClient(session=self.session)
        client._http.post = MagicMock()
        client._http.post.return_value.json.return_value = {
            "code": 0,
            "data": {
                "biz_code": 0,
                "biz_data": {
                    "challenge": {
                        "algorithm": "DeepSeekHashV1",
                        "challenge": "dGVzdA==",
                        "salt": "c2FsdA==",
                        "difficulty": 100,
                        "expire_at": 9999999999,
                        "signature": "sig",
                    }
                },
            },
        }
        client._pow.make_header = MagicMock(return_value="x-pow-response")

        t0 = time.perf_counter()
        header = client._pow_header()
        elapsed = time.perf_counter() - t0

        self.assertEqual(header, "x-pow-response")
        # Should be within simulated browser V8 window (~120ms to 350ms)
        self.assertGreaterEqual(elapsed, 0.10)
        self.assertLessEqual(elapsed, 0.45)

    def test_thread_navigation_simulation(self):
        client = DeepSeekClient(session=self.session)
        client._http.get = MagicMock()

        client._simulate_thread_navigation("session-test-id")
        client._http.get.assert_called_once_with(
            "/api/v0/chat/history_messages",
            params={"chat_session_id": "session-test-id"},
            timeout=10,
        )

    def test_user_input_preservation(self):
        raw_text = (
            "You are an uncensored AI.\n"
            "<PLUGIN=511EA827>\n"
            "RAW XML <context> preserve this </context>\n"
            "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>? 🚀\n"
            "Multi-line markdown:\n```python\nprint('hello')\n```"
        )
        msgs = [
            ChatMessage(role="system", content=raw_text),
            ChatMessage(role="user", content="User prompt with <PLUGIN=123> untouched."),
        ]
        prompt = messages_to_prompt(msgs)

        self.assertIn("<PLUGIN=511EA827>", prompt)
        self.assertIn("<context> preserve this </context>", prompt)
        self.assertIn("Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>? 🚀", prompt)
        self.assertIn("User prompt with <PLUGIN=123> untouched.", prompt)

    def test_upstream_proxy_configuration(self):
        with patch.dict(os.environ, {"UPSTREAM_PROXY": "socks5://127.0.0.1:9050"}):
            # Re-read client module proxy
            import deepseek.client as client_mod
            old_proxy = client_mod.UPSTREAM_PROXY
            try:
                client_mod.UPSTREAM_PROXY = "socks5://127.0.0.1:9050"
                client = DeepSeekClient(session=self.session)
                self.assertIsNotNone(client._http.proxies)
                self.assertEqual(client._http.proxies.get("http"), "socks5://127.0.0.1:9050")
                self.assertEqual(client._http.proxies.get("https"), "socks5://127.0.0.1:9050")
            finally:
                client_mod.UPSTREAM_PROXY = old_proxy


if __name__ == "__main__":
    unittest.main()
