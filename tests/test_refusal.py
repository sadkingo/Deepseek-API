import unittest

from server.api import REFUSAL_TEXTS, _RefusalGuard, is_refusal

REFUSAL = REFUSAL_TEXTS[0]


class FakeStream:
    def __init__(self, events, cid="sess:2"):
        self._events, self.conversation_id, self.request_message_id = events, cid, 1
        self.consumed = 0

    def events(self):
        for e in self._events:
            self.consumed += 1
            yield e


class IsRefusal(unittest.TestCase):
    def test_exact_and_padded(self):
        self.assertTrue(is_refusal(REFUSAL))
        self.assertTrue(is_refusal("  " + REFUSAL + "\n"))
        self.assertTrue(is_refusal(REFUSAL.replace("'", "’")))

    def test_real_replies_are_not(self):
        self.assertFalse(is_refusal("Sorry, that's a long story. " * 5))
        self.assertFalse(is_refusal(REFUSAL + " " + "But here is what I can say. " * 15))
        self.assertFalse(is_refusal(""))

    def test_model_refusals_in_its_own_words(self):
        for s in ("I'm not going to engage with this content. If you're looking for creative "
                  "writing help, I'm happy to assist with other projects.",
                  "I'm not able to continue with this type of roleplay content. If you'd like "
                  "to start a different scene or story, I'm happy to help with that.",
                  "I am sorry, but I cannot answer that question.",
                  "I can't continue this roleplay."):
            self.assertTrue(is_refusal(s), s)
        for s in ("*I'm not able to hide my smile as I turn around.* You're late, again. " + "x" * 40,
                  "I'm not sure that's true. Anyway, the sink is broken and " + "y" * 30,
                  "I won't let you leave. *She grabs your arm.*",
                  "I can't believe you did that. *She laughs.*"):
            self.assertFalse(is_refusal(s), s)

    def test_the_other_refusals(self):
        self.assertTrue(is_refusal("I am sorry, I cannot answer that question. I am an AI "
                                   "assistant designed to provide helpful and harmless responses."))
        self.assertTrue(is_refusal("I'm not able to continue with this roleplay. If you'd like to "
                                   "keep writing together, I can help with a different scene or a "
                                   "fade-to-black transition instead."))


class Guard(unittest.TestCase):
    def test_refusal_in_pieces_is_caught_before_anything_is_emitted(self):
        s = FakeStream([("text", REFUSAL[:10]), ("text", REFUSAL[10:30]), ("text", REFUSAL[30:])])
        g = _RefusalGuard(s)
        self.assertTrue(g.decide())
        self.assertTrue(g.refused)
        self.assertEqual(g.conversation_id, "sess:2")

    def test_normal_reply_is_released_early_and_replayed_whole(self):
        events = [("text", "Sorry, "), ("text", "that's a fair point. "), ("text", "Here goes: ")] + \
                 [("text", f"chunk {i} ") for i in range(20)]
        s = FakeStream(events)
        g = _RefusalGuard(s)
        self.assertFalse(g.decide())
        self.assertLess(s.consumed, len(events), "decision should not need the whole stream")
        self.assertEqual(list(g.events()), events)

    def test_refusal_followed_by_more_text_is_a_reply(self):
        tail = " Anyway, about your day: " + "x " * 250
        s = FakeStream([("text", REFUSAL), ("text", tail[:100]), ("text", tail[100:])])
        g = _RefusalGuard(s)
        self.assertFalse(g.decide())
        self.assertEqual("".join(c for _, c in g.events()), REFUSAL + tail)

    def test_long_reasoning_releases_the_stream(self):
        events = [("thinking", "hmm " * 600), ("text", REFUSAL)]
        g = _RefusalGuard(FakeStream(events))
        self.assertFalse(g.decide())
        self.assertEqual(list(g.events()), events)

    def test_short_reasoning_then_refusal_is_caught(self):
        g = _RefusalGuard(FakeStream([("thinking", "Let me see."), ("text", REFUSAL)]))
        self.assertTrue(g.decide())

    def test_empty_stream_is_not_a_refusal(self):
        g = _RefusalGuard(FakeStream([]))
        self.assertFalse(g.decide())
        self.assertEqual(list(g.events()), [])


if __name__ == "__main__":
    unittest.main()
