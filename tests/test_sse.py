import unittest

from deepseek.client import UpstreamGaveUp, _parse_sse


def _stream(*lines):
    return [line.encode() for line in lines]


class ParseSseTests(unittest.TestCase):
    def test_reply_text_is_yielded(self):
        lines = _stream(
            "event: ready",
            'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}',
            ": ",
            'data: {"v":{"response":{"message_id":2,"fragments":[{"type":"RESPONSE","content":"Hel"}]}}}',
            'data: {"v":"lo"}',
        )
        meta = {}
        out = list(_parse_sse(lines, meta))
        self.assertEqual(out, [("text", "Hel"), ("text", "lo")])
        self.assertEqual(meta["message_id"], 2)

    def test_error_hint_raises_instead_of_looking_empty(self):
        # What DeepSeek sends when it accepts a request, heartbeats for ~60s,
        # then abandons it (observed 2026-09-14).
        lines = _stream(
            "event: ready",
            'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}',
            ": ",
            ": ",
            "event: hint",
            'data: {"type":"error","content":"Server busy, please try again later.",'
            '"clear_response":true,"finish_reason":"generation_timeout"}',
            "event: close",
            'data: {"click_behavior":"retry","auto_resume":false}',
        )
        with self.assertRaises(UpstreamGaveUp) as ctx:
            list(_parse_sse(lines, {}))
        self.assertIn("Server busy", str(ctx.exception))
        self.assertIn("generation_timeout", str(ctx.exception))

    def test_harmless_hint_is_ignored(self):
        lines = _stream(
            "event: hint",
            'data: {"type":"notice","content":"whatever"}',
            'data: {"v":{"response":{"message_id":2,"fragments":[{"type":"RESPONSE","content":"ok"}]}}}',
        )
        self.assertEqual(list(_parse_sse(lines, {})), [("text", "ok")])

    def test_empty_stream_is_logged_with_its_frames(self):
        lines = _stream(
            "event: ready",
            'data: {"request_message_id":1,"response_message_id":2}',
            ": ",
            "event: close",
            'data: {"click_behavior":"retry"}',
        )
        with self.assertLogs("deepseek.upstream", level="WARNING") as logs:
            self.assertEqual(list(_parse_sse(lines, {})), [])
        self.assertTrue(any("closed without content" in m for m in logs.output))
        self.assertTrue(any("event: close" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
