import unittest

from deepseek.client import (COMPLETION_PATH, REGENERATE_PATH, EditRateLimited,
                             UpstreamGaveUp, _EditQuota, _Stream)


class FakeClient:
    """Only what `_Stream._attempt` touches."""

    def __init__(self):
        self._edit_quota = _EditQuota()
        self.sessions_created = 0

    def create_chat_session(self):
        self.sessions_created += 1
        return "new-session"

    def _simulate_thread_navigation(self, sid):
        return []

    @staticmethod
    def _refused_by_moderation(messages, message_id):
        return False


def stream_with(client, **kw):
    kw.setdefault("prompt", "hello again")
    kw.setdefault("session_id", "sess")
    kw.setdefault("parent_id", 4)
    kw.setdefault("model", None)
    kw.setdefault("thinking", False)
    kw.setdefault("search", False)
    return _Stream(client, kw["prompt"], kw["session_id"], kw["parent_id"],
                   kw["model"], kw["thinking"], kw["search"], None,
                   kw.get("regenerate_of"))


class RegenerateFallback(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.calls = []

    def _request(self, fail_regenerate):
        def request(path, body, meta):
            self.calls.append((path, body))
            if path == REGENERATE_PATH and fail_regenerate:
                raise EditRateLimited(
                    "DeepSeek gave up on the request (regeneration_rate_limit): "
                    "Editing/regeneration too frequently. Try again later.",
                    reason="regeneration_rate_limit",
                    content="Editing/regeneration too frequently. Try again later.")
            meta["message_id"] = 9
            yield ("text", "a reply")
        return request

    def test_quota_error_becomes_a_branch(self):
        s = stream_with(self.client, regenerate_of=6)
        s._request = self._request(fail_regenerate=True)
        with self.assertLogs("deepseek.upstream", level="WARNING"):
            self.assertEqual(list(s._attempt({})), [("text", "a reply")])
        self.assertEqual([p for p, _ in self.calls], [REGENERATE_PATH, COMPLETION_PATH])
        branch = self.calls[1][1]
        self.assertEqual(branch["parent_message_id"], 4)   # same state as the swipe
        self.assertEqual(branch["prompt"], "hello again")
        self.assertEqual(self.client.sessions_created, 0)  # stays in the chat
        self.assertTrue(self.client._edit_quota.spent())

    def test_the_endpoint_is_skipped_while_the_quota_is_spent(self):
        self.client._edit_quota.note()
        s = stream_with(self.client, regenerate_of=6)
        s._request = self._request(fail_regenerate=True)
        with self.assertLogs("deepseek.upstream", level="INFO"):
            self.assertEqual(list(s._attempt({})), [("text", "a reply")])
        self.assertEqual([p for p, _ in self.calls], [COMPLETION_PATH])

    def test_a_working_regeneration_is_left_alone(self):
        s = stream_with(self.client, regenerate_of=6)
        s._request = self._request(fail_regenerate=False)
        self.assertEqual(list(s._attempt({})), [("text", "a reply")])
        self.assertEqual([p for p, _ in self.calls], [REGENERATE_PATH])
        self.assertEqual(self.calls[0][1]["child_message_id"], 6)
        self.assertFalse(self.client._edit_quota.spent())

    def test_a_quota_error_after_text_is_not_swallowed(self):
        def request(path, body, meta):
            self.calls.append((path, body))
            yield ("text", "half a repl")
            raise EditRateLimited("too frequently", reason="regeneration_rate_limit")
        s = stream_with(self.client, regenerate_of=6)
        s._request = request
        with self.assertRaises(EditRateLimited):
            list(s._attempt({}))
        self.assertEqual([p for p, _ in self.calls], [REGENERATE_PATH])


class RegenerationLimit(unittest.TestCase):
    """DeepSeek caps how often ONE message is regenerated ("Regeneration limit
    reached" after six). The swipe then edits the message - same text plus a
    space - which is a new branch with its own allowance."""

    def setUp(self):
        self.client = FakeClient()
        self.calls = []

    def request(self, path, body, meta):
        self.calls.append((path, dict(body)))
        if path == REGENERATE_PATH:
            raise UpstreamGaveUp(
                "DeepSeek gave up on the request (regeneration_limit): "
                "Regeneration limit reached",
                reason="regeneration_limit", content="Regeneration limit reached")
        meta["message_id"] = 28
        yield ("text", "a fresh reply")

    def test_the_limit_becomes_an_edit_with_a_space_added(self):
        s = stream_with(self.client, regenerate_of=24, parent_id=11,
                        prompt="sadking: *few days after*")
        s._request = self.request
        with self.assertLogs("deepseek.upstream", level="WARNING") as logs:
            self.assertEqual(list(s._attempt({})), [("text", "a fresh reply")])
        self.assertEqual([p for p, _ in self.calls], [REGENERATE_PATH, COMPLETION_PATH])
        edit = self.calls[1][1]
        self.assertEqual(edit["parent_message_id"], 11)       # same state as the swipe
        self.assertEqual(edit["prompt"], "sadking: *few days after* ")
        self.assertEqual(self.client.sessions_created, 0)     # same chat
        # Only this message is capped; the account's quota is untouched.
        self.assertFalse(self.client._edit_quota.spent())
        self.assertIn("regeneration limit", logs.output[0])

    def test_other_give_ups_still_fail(self):
        def request(path, body, meta):
            raise UpstreamGaveUp("busy", reason="generation_timeout", content="Server busy")
            yield
        s = stream_with(self.client, regenerate_of=24)
        s._request = request
        with self.assertRaises(UpstreamGaveUp):
            list(s._attempt({}))


class Cooldown(unittest.TestCase):
    def test_note_and_clear(self):
        q = _EditQuota()
        self.assertFalse(q.spent())
        q.note(cooldown=60)
        self.assertTrue(q.spent())
        q.note(cooldown=0.0)      # never shortens an existing cooldown
        self.assertTrue(q.spent())
        q.clear()
        self.assertFalse(q.spent())

    def test_an_elapsed_cooldown_is_over(self):
        q = _EditQuota()
        q.note(cooldown=-1)
        self.assertFalse(q.spent())


if __name__ == "__main__":
    unittest.main()


class ReplyTexts(unittest.TestCase):
    def test_only_response_fragments_of_assistant_messages(self):
        from deepseek.client import DeepSeekClient
        client = DeepSeekClient.__new__(DeepSeekClient)   # no session needed
        client._simulate_thread_navigation = lambda sid: [
            {"message_id": 1, "role": "USER",
             "fragments": [{"type": "REQUEST", "content": "hi"}]},
            {"message_id": 2, "role": "ASSISTANT",
             "fragments": [{"type": "THINK", "content": "hmm"},
                           {"type": "RESPONSE", "content": "Hello "},
                           {"type": "RESPONSE", "content": "there."},
                           {"type": "TIP", "content": "AI-generated"}]},
        ]
        self.assertEqual(client.reply_texts("s"), {2: "Hello there."})
        client._simulate_thread_navigation = lambda sid: None
        self.assertIsNone(client.reply_texts("s"))
