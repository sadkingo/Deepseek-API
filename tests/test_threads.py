import time
import unittest

from server.threads import (EDIT_SIMILARITY, RECENT_EDIT_SIMILARITY, TurnIndex,
                            gist, similarity)

SYS = ("system", "You are Allison, a stern but caring wife. " * 20, "")
GREETING = ("assistant", "*Allison looks up from the stove.* You're late.", "")
NOTE = "\nSYSTEM NOTE: Do not include the following words: god, worship"


def hist(*turns):
    """system + '.' + greeting, then the given (role, text) pairs."""
    h = [SYS, ("user", ".", ""), GREETING]
    h += [(r, t, "") for r, t in turns]
    return h


class NewChatVersusEditedFirstTurn(unittest.TestCase):
    """The first real turn after the greeting: which chat does it belong to?"""

    def setUp(self):
        self.idx = TurnIndex(path=None)
        self.first = hist(("user", "sadking: *I sit down quietly* sorry, traffic." + NOTE))
        self.idx.remember(self.first, "sessA:2", None, "", "Allison sighs.", "acct")

    def test_same_first_message_again_is_a_regeneration_in_the_same_chat(self):
        m = self.idx.find(self.first, "acct")
        self.assertIsNotNone(m)
        self.assertTrue(m.sibling)
        self.assertEqual(m.cid, "sessA")
        self.assertEqual(m.kind, "regenerate")
        self.assertEqual(m.regenerate_of, 2)

    def test_same_message_without_the_note_still_regenerates(self):
        h = hist(("user", "sadking: *I sit down quietly* sorry, traffic."))
        m = self.idx.find(h, "acct")
        self.assertEqual((m.kind, m.cid), ("regenerate", "sessA"))

    def test_rewritten_first_message_is_an_edit_of_that_chat(self):
        h = hist(("user", "sadking: *I sit down quietly* sorry, the traffic was awful." + NOTE))
        m = self.idx.find(h, "acct")
        self.assertIsNotNone(m)
        self.assertEqual((m.kind, m.cid, m.sibling), ("edit", "sessA", True))
        self.assertIsNone(m.regenerate_of)
        self.assertGreater(m.similarity, EDIT_SIMILARITY)

    def test_different_first_message_is_a_new_chat(self):
        h = hist(("user", "sadking: *I slam the door and storm past her without a word*" + NOTE))
        self.assertIsNone(self.idx.find(h, "acct"))
        self.assertIn("new conversation", self.idx.last_miss())
        self.assertIn("sessA:2", self.idx.last_miss())

    def test_notes_do_not_make_short_messages_look_alike(self):
        idx = TurnIndex(path=None)
        idx.remember(hist(("user", "hi" + NOTE)), "sessB:2", None, "", "Hello.", "acct")
        self.assertIsNone(idx.find(hist(("user", "run!" + NOTE)), "acct"))

    def test_old_records_without_text_only_match_by_shape(self):
        # A record from before the gist was stored: "" text.
        idx = TurnIndex(path=None)
        idx.remember(self.first, "sessC:2", None, "", "Allison sighs.", "acct")
        for t in idx._turns.values():
            t.text = ""
        self.assertEqual(idx.find(self.first, "acct").kind, "regenerate")
        h = hist(("user", "sadking: *I sit down quietly* sorry, the traffic was awful." + NOTE))
        self.assertIsNone(idx.find(h, "acct"))

    def test_a_recent_chat_accepts_a_looser_rewrite(self):
        h = hist(("user", "sadking: *I stay silent and sit*" + NOTE))
        sim = similarity(gist(h[-1][1]), gist(self.first[-1][1]))
        self.assertTrue(RECENT_EDIT_SIMILARITY <= sim < EDIT_SIMILARITY,
                        f"test needs a middling similarity, got {sim:.2f}")
        self.assertEqual(self.idx.find(h, "acct").kind, "edit")
        for t in self.idx._turns.values():
            t.ts = time.time() - 3600
        self.assertIsNone(self.idx.find(h, "acct"))


class ContinuingAChat(unittest.TestCase):
    def setUp(self):
        self.idx = TurnIndex(path=None)
        self.h1 = hist(("user", "sadking: hello there" + NOTE))
        self.idx.remember(self.h1, "sess:2", None, "", "Allison: Hello yourself.", "acct")
        self.h2 = hist(("user", "sadking: hello there"),
                       ("assistant", "Allison: Hello yourself."),
                       ("user", "sadking: how was your day?" + NOTE))
        self.idx.remember(self.h2, "sess:4", "sess:2", "", "Allison: Long.", "acct")

    def test_next_message_appends(self):
        h = self.h2[:-1] + [("user", "sadking: how was your day?", ""),
                            ("assistant", "Allison: Long.", ""),
                            ("user", "sadking: tell me about it" + NOTE, "")]
        m = self.idx.find(h, "acct")
        self.assertEqual((m.cid, m.kind, m.resume_from), ("sess:4", "append", len(h) - 1))

    def test_same_question_again_regenerates_that_answer(self):
        m = self.idx.find(self.h2, "acct")
        self.assertEqual((m.cid, m.kind, m.regenerate_of), ("sess:2", "regenerate", 4))

    def test_changed_question_is_an_edit_branch(self):
        h = self.h2[:-1] + [("user", "sadking: how was work today?" + NOTE, "")]
        m = self.idx.find(h, "acct")
        self.assertEqual((m.cid, m.kind, m.regenerate_of), ("sess:2", "edit", None))
        self.assertGreater(m.similarity, 0)

    def test_continue_is_continue(self):
        h = self.h2[:-1] + [("user", "sadking: how was your day?", ""),
                            ("assistant", "Allison: Long.", "")]
        m = self.idx.find(h, "acct")
        self.assertEqual((m.cid, m.kind, m.continues), ("sess:4", "continue", True))

    def test_user_message_id_is_kept(self):
        self.idx.remember(self.h2, "sess:6", "sess:2", "", "Allison: Meh.", "acct",
                          user_mid=5)
        self.assertEqual(self.idx._turns["sess:6"].user_mid, 5)
        self.assertEqual(self.idx._turns["sess:6"].message_id, 6)


class Gist(unittest.TestCase):
    def test_gist_drops_notes_and_normalises(self):
        self.assertEqual(gist("hello   there\n\nSYSTEM NOTE: no swearing"), "hello there")

    def test_similarity_bounds(self):
        self.assertEqual(similarity("", "abc"), 0.0)
        self.assertEqual(similarity("abc", "abc"), 1.0)
        self.assertLess(similarity("a completely different thing", "abc"), 0.3)


if __name__ == "__main__":
    unittest.main()


class RefusedStates(unittest.TestCase):
    def test_next_turn_after_a_refusal_edits_the_first_message(self):
        idx = TurnIndex(path=None)
        h1 = hist(("user", "sadking: hello there" + NOTE))
        idx.remember(h1, "sess:2", None, "", "Allison: Hello yourself.", "acct")
        h2 = hist(("user", "sadking: hello there"), ("assistant", "Allison: Hello yourself."),
                  ("user", "sadking: come closer" + NOTE))
        idx.remember(h2, "sess:4", "sess:2", "", "I'm not going to engage with this content.",
                     "acct", poisoned=True)
        h3 = h2[:-1] + [("user", "sadking: come closer", ""),
                        ("assistant", "I'm not going to engage with this content.", ""),
                        ("user", "sadking: *I sit beside her*" + NOTE, "")]
        m = idx.find(h3, "acct")
        self.assertIsNotNone(m)
        self.assertTrue(m.recovering)
        self.assertEqual((m.cid, m.resume_from, m.sibling, m.kind), ("sess", 0, True, "edit"))
        self.assertIn("refusal", m.describe())

    def test_a_swipe_on_a_refusal_regenerates_from_the_good_state(self):
        idx = TurnIndex(path=None)
        h1 = hist(("user", "sadking: hello there" + NOTE))
        idx.remember(h1, "sess:2", None, "", "Allison: Hello yourself.", "acct")
        h2 = hist(("user", "sadking: hello there"), ("assistant", "Allison: Hello yourself."),
                  ("user", "sadking: come closer" + NOTE))
        idx.remember(h2, "sess:4", "sess:2", "", "I'm not going to engage.", "acct", poisoned=True)
        m = idx.find(h2, "acct")
        self.assertFalse(m.recovering)
        self.assertEqual((m.cid, m.kind, m.regenerate_of), ("sess:2", "regenerate", 4))
