import unittest
from unittest.mock import MagicMock
from server.ratelimit import RateLimiter, _client_key


class TestRateLimiter(unittest.TestCase):
    def test_sliding_window_allows_within_limit(self):
        limiter = RateLimiter(limit=3, window=10.0)
        t0 = 1000.0
        allowed1, remaining1, _ = limiter.hit("ip1", t0)
        self.assertTrue(allowed1)
        self.assertEqual(remaining1, 2)

        allowed2, remaining2, _ = limiter.hit("ip1", t0 + 1.0)
        self.assertTrue(allowed2)
        self.assertEqual(remaining2, 1)

        allowed3, remaining3, _ = limiter.hit("ip1", t0 + 2.0)
        self.assertTrue(allowed3)
        self.assertEqual(remaining3, 0)

        # 4th hit within window should be rejected
        allowed4, remaining4, retry_after = limiter.hit("ip1", t0 + 3.0)
        self.assertFalse(allowed4)
        self.assertEqual(remaining4, 0)
        self.assertGreater(retry_after, 0)

        # After window expires, allowed again
        allowed5, remaining5, _ = limiter.hit("ip1", t0 + 11.0)
        self.assertTrue(allowed5)

    def test_client_key_trusted_proxy(self):
        # Peer IP is 127.0.0.1 (trusted proxy default)
        req = MagicMock()
        req.client.host = "127.0.0.1"
        req.headers = {"x-forwarded-for": "203.0.113.195, 10.0.0.1"}
        self.assertEqual(_client_key(req), "203.0.113.195")

        # Untrusted remote peer: X-Forwarded-For should be ignored
        req_untrusted = MagicMock()
        req_untrusted.client.host = "198.51.100.50"
        req_untrusted.headers = {"x-forwarded-for": "1.2.3.4"}
        self.assertEqual(_client_key(req_untrusted), "198.51.100.50")


if __name__ == "__main__":
    unittest.main()
