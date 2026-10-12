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


class EditedReplies(unittest.TestCase):
    """The user rewrites a reply in the frontend, then goes on. DeepSeek's thread
    still holds the reply as generated, so the turn is picked up from before it
    and the edited reply is resent (the case behind this: JanitorAI sent back a
    1,878-character reply that DeepSeek had generated as 2,917 characters, with
    the same opening, and the old code resumed after it as if nothing changed)."""

    HEADER = "`Time: 19:26 | October 16, Friday | Location: Rossi's Trattoria`"
    R2 = HEADER + "\n\n*She stayed.* " + "The evening went on and on. " * 40
    R2_EDITED = HEADER + "\n\n*She left before dessert.*"

    def setUp(self):
        self.idx = TurnIndex(path=None)
        self.h1 = hist(("user", "sadking: few days after, a date" + NOTE))
        self.idx.remember(self.h1, "sess:2", None, "", "Jillian walks in.", "acct")
        self.h2 = hist(("user", "sadking: few days after, a date"),
                       ("assistant", "Jillian walks in."),
                       ("user", "sadking: you see where you stand?" + NOTE))
        self.idx.remember(self.h2, "sess:4", "sess:2", "", self.R2, "acct")

    def after_edit(self, last):
        return hist(("user", "sadking: few days after, a date"),
                    ("assistant", "Jillian walks in."),
                    ("user", "sadking: you see where you stand?"),
                    ("assistant", self.R2_EDITED),
                    ("user", last))

    def test_an_edited_reply_is_not_continued(self):
        # Same opening, so the old 160-character check called it kept.
        self.assertEqual(self.R2[:60], self.R2_EDITED[:60])
        h = self.after_edit("sadking: move on" + NOTE)
        m = self.idx.find(h, "acct")
        # The state before the edited turn; the question, the edited reply and
        # the new message are all resent from there.
        self.assertEqual((m.cid, m.kind), ("sess:2", "append"))
        self.assertEqual(m.resume_from, 5)  # the question the edited reply answered
        self.assertTrue(m.reply_edited)
        self.assertIn("edited in the client", m.describe())

    def test_the_unedited_reply_still_resumes_as_before(self):
        h = hist(("user", "sadking: few days after, a date"),
                 ("assistant", "Jillian walks in."),
                 ("user", "sadking: you see where you stand?"),
                 ("assistant", "  " + self.R2.replace(" ", "  ") + "\n"),  # whitespace only
                 ("user", "sadking: move on" + NOTE))
        m = self.idx.find(h, "acct")
        self.assertEqual((m.cid, m.resume_from, m.reply_edited), ("sess:4", len(h) - 1, False))

    def test_a_changed_header_alone_is_an_edit(self):
        h = self.after_edit("sadking: move on" + NOTE)
        h[6] = ("assistant", self.R2.replace("19:26", "21:00"), "")
        self.assertEqual(self.idx.find(h, "acct").cid, "sess:2")

    def test_after_the_resend_the_chat_carries_on_normally(self):
        h3 = self.after_edit("sadking: move on" + NOTE)
        m = self.idx.find(h3, "acct")
        self.idx.remember(h3, "sess:6", m.cid, "", "Jillian goes quiet.", "acct",
                          carried_from=m.resume_from)
        # Next turn: only the newest message is new.
        h4 = self.after_edit("sadking: move on") + [
            ("assistant", "Jillian goes quiet.", ""),
            ("user", "sadking: bye" + NOTE, "")]
        m4 = self.idx.find(h4, "acct")
        self.assertEqual((m4.cid, m4.kind, m4.resume_from), ("sess:6", "append", len(h4) - 1))
        self.assertFalse(m4.reply_edited)
        # A swipe on the resent turn is DeepSeek's Regenerate of its answer.
        again = self.idx.find(h3, "acct")
        self.assertEqual((again.cid, again.kind, again.regenerate_of),
                         ("sess:2", "regenerate", 6))

    def test_an_edited_older_reply_is_caught_too(self):
        h = hist(("user", "sadking: few days after, a date"),
                 ("assistant", "Jillian walks in, soaked."),   # edited
                 ("user", "sadking: you see where you stand?"),
                 ("assistant", self.R2),
                 ("user", "sadking: move on" + NOTE))
        # Both states sit after the edited first reply: nothing can be resumed,
        # so the history goes out whole.
        self.assertIsNone(self.idx.find(h, "acct"))
        self.assertIn("edited", self.idx.last_miss())

    def test_records_from_before_reply_hashes_are_never_edited(self):
        self.idx._turns["sess:4"].reply_hash = ""
        h = self.after_edit("sadking: move on" + NOTE)
        self.assertEqual(self.idx.find(h, "acct").cid, "sess:4")


class EveryReplyIsChecked(unittest.TestCase):
    """Edits are caught wherever the text went to DeepSeek: as our reply, inside
    a chat's first prompt (the greeting), or resent after an earlier edit - and
    for records made before reply hashes existed, once they are filled in from
    DeepSeek's copy of the chat."""

    def build(self, legacy=False):
        idx = TurnIndex(path=None)
        h1 = hist(("user", "sadking: hi" + NOTE))
        idx.remember(h1, "sess:2", None, "", "Jillian waves.", "acct")
        h2 = hist(("user", "sadking: hi"), ("assistant", "Jillian waves."),
                  ("user", "sadking: date?" + NOTE))
        idx.remember(h2, "sess:4", "sess:2", "", "She says yes, then leaves.", "acct")
        h3 = hist(("user", "sadking: hi"), ("assistant", "Jillian waves."),
                  ("user", "sadking: date?"), ("assistant", "She says yes, then leaves."),
                  ("user", "sadking: cheater" + NOTE))
        idx.remember(h3, "sess:6", "sess:4", "", "Jillian flinches.", "acct")
        if legacy:  # as recorded before this existed
            for t in idx._turns.values():
                t.reply_hash, t.carried_replies = "", None
        return idx

    def edited_history(self):
        return hist(("user", "sadking: hi"), ("assistant", "Jillian waves."),
                    ("user", "sadking: date?"), ("assistant", "She says yes."),  # edited
                    ("user", "sadking: cheater"), ("assistant", "Jillian flinches."),
                    ("user", "sadking: bye" + NOTE))

    def test_old_records_are_checked_once_filled_in_from_deepseek(self):
        idx = self.build(legacy=True)
        h = self.edited_history()
        # Without the reply text the edit is invisible: the newest state wins.
        self.assertEqual(idx.find(h, "acct").cid, "sess:6")
        self.assertEqual(sorted(idx.unhashed("sess:6")), [2, 4, 6])
        filled = idx.fill_reply_hashes("sess", {2: "Jillian waves.",
                                               4: "She says yes, then leaves.",
                                               6: "Jillian flinches."})
        self.assertEqual(filled, 3)
        self.assertEqual(idx.unhashed("sess:6"), [])
        m = idx.find(h, "acct")
        self.assertEqual((m.cid, m.resume_from, m.reply_edited), ("sess:2", 5, True))

    def test_a_reply_deepseek_no_longer_has_is_asked_for_once(self):
        idx = self.build(legacy=True)
        idx.fill_reply_hashes("sess", {2: "Jillian waves."})
        self.assertEqual(idx.unhashed("sess:6"), [])           # not fetched again
        self.assertEqual(idx.find(self.edited_history(), "acct").cid, "sess:6")

    def test_an_edited_greeting_means_the_chat_cannot_be_resumed(self):
        idx = self.build()
        h = self.edited_history()
        h[2] = ("assistant", "*Jillian looks up.* You're early.", "")  # the greeting
        self.assertIsNone(idx.find(h, "acct"))
        self.assertIn("edited", idx.last_miss())

    def test_a_resent_reply_edited_again_is_caught(self):
        idx = self.build()
        h = self.edited_history()
        m = idx.find(h, "acct")
        idx.remember(h, "sess:8", m.cid, "", "Jillian walks out.", "acct",
                     carried_from=m.resume_from)
        h2 = list(h[:-1]) + [("user", "sadking: bye", ""),
                             ("assistant", "Jillian walks out.", ""),
                             ("user", "sadking: wait" + NOTE, "")]
        self.assertEqual(idx.find(h2, "acct").cid, "sess:8")    # carried as sent
        h2[6] = ("assistant", "She says no.", "")               # edited once more
        m2 = idx.find(h2, "acct")
        self.assertEqual((m2.cid, m2.reply_edited), ("sess:2", True))

    def test_a_trimmed_history_is_not_an_edit(self):
        idx = self.build()
        h = hist(("user", "sadking: hi"), ("assistant", "Jillian waves."),
                 ("user", "sadking: date?"), ("assistant", "She says yes, then leaves."),
                 ("user", "sadking: cheater"), ("assistant", "Jillian flinches."),
                 ("user", "sadking: bye" + NOTE))
        trimmed = [h[0]] + h[4:]   # the context window dropped the oldest turns
        m = idx.find(trimmed, "acct")
        self.assertEqual((m.cid, m.reply_edited), ("sess:6", False))

    def test_keeping_one_swipe_is_not_an_edit(self):
        idx = self.build()
        h2 = hist(("user", "sadking: hi"), ("assistant", "Jillian waves."),
                  ("user", "sadking: date?" + NOTE))
        idx.remember(h2, "sess:5", "sess:2", "", "She says no.", "acct")  # a swipe
        h = hist(("user", "sadking: hi"), ("assistant", "Jillian waves."),
                 ("user", "sadking: date?"), ("assistant", "She says no."),
                 ("user", "sadking: why" + NOTE))
        m = idx.find(h, "acct")
        self.assertEqual((m.cid, m.reply_edited), ("sess:5", False))
