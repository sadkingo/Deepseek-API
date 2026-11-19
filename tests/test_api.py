import unittest
from fastapi.testclient import TestClient
from server.api import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz(self):
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_list_models(self):
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["object"], "list")
        ids = [m["id"] for m in data["data"]]
        self.assertIn("deepseek-chat", ids)
        self.assertIn("deepseek-reasoner", ids)

    def test_get_single_model(self):
        res = self.client.get("/v1/models/deepseek-chat")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], "deepseek-chat")

        res_404 = self.client.get("/v1/models/non-existent-model")
        self.assertEqual(res_404.status_code, 404)

    def test_embeddings_unsupported(self):
        res = self.client.post("/v1/embeddings", json={"input": "test"})
        self.assertEqual(res.status_code, 501)


if __name__ == "__main__":
    unittest.main()


class SystemPromptResync(unittest.TestCase):
    """A chat whose system prompt changed (the user rewrote their instructions)
    is resent from its first message; DeepSeek only saw the old prompt."""

    def setUp(self):
        import asyncio
        from unittest import mock
        from server import api
        from server.threads import TurnIndex
        self.api, self.run = api, asyncio.run
        self.idx = TurnIndex(path=None)
        self.old = [("system", "OLD RULES. Write well.", ""), ("user", "hi", "")]
        self.new = [("system", "NEW RULES. Write out directions.", ""), ("user", "hi", "")]
        self.idx.remember(self.old + [("assistant", "Hello.", ""), ("user", "go", "")],
                          "sess:2", None, "", "Going.", "acct")
        self.first = "OLD RULES. Write well.\n\nhi"

        class FakeClient:
            calls = 0
            def first_prompt(inner, session, mid):
                FakeClient.calls += 1
                return self.first
        self.client = FakeClient()
        for p in (mock.patch.object(api, "_threads", self.idx),
                  mock.patch.object(api, "_resynced_at", {}),
                  mock.patch.object(api.debuglog, "log_thread", lambda *a, **k: None)):
            p.start()
            self.addCleanup(p.stop)

    def changed(self, history):
        return self.run(self.api._system_changed(self.client, "sess:2", history))

    def test_same_prompt_is_left_alone(self):
        self.assertFalse(self.changed(self.old))

    def test_changed_prompt_resyncs_once_per_cooldown(self):
        self.assertTrue(self.changed(self.new))
        self.assertFalse(self.changed(self.new))   # again right away: cooldown

    def test_older_records_are_checked_against_deepseeks_first_prompt(self):
        self.idx._turns["sess:2"].sys_hash = ""
        self.assertFalse(self.changed(self.old))   # DeepSeek has these rules
        self.assertEqual(self.client.calls, 1)
        self.assertFalse(self.changed(self.old))   # now recorded: no second read
        self.assertEqual(self.client.calls, 1)

    def test_older_records_with_a_since_changed_prompt_resync(self):
        self.idx._turns["sess:2"].sys_hash = ""
        self.assertTrue(self.changed(self.new))


class PostHistoryInstructions(unittest.TestCase):
    """A block of the system prompt marked [POST-HISTORY] is sent after the
    newest message on every turn, not at the top: a DeepSeek thread sees the
    top of a chat only once (2026-11-01: the user's enforcement block sat 40k
    characters above the message it was about, and the model kept skipping the
    narration it was meant to protect)."""

    SYSTEM = ("RULES. Write well.\n[POST-HISTORY]\nWrite out every direction in "
              "sadking's message first, in full.\n[/POST-HISTORY]\nCARD: Nikki.")

    def setUp(self):
        from unittest import mock
        from server import api
        from server.threads import TurnIndex
        from deepseek.client import Reply
        self.prompts = []
        test = self

        class FakeClient:
            session = type("S", (), {"token": "tok"})()
            def chat(self, prompt, conversation_id, *a, **k):
                test.prompts.append((prompt, conversation_id))
                n = len(test.prompts) * 2
                return Reply(text=f"Reply {n}." + ("\nWrite out every direction in "
                             "sadking's message first, in full." if n == 2 else ""),
                             conversation_id=f"sess:{n}", request_message_id=n - 1)
            def pace_hint(self):
                return 0
        for p in (mock.patch.object(api, "get_client", lambda *a, **k: FakeClient()),
                  mock.patch.object(api, "_threads", TurnIndex(path=None)),
                  mock.patch.object(api.debuglog, "log_thread", lambda *a, **k: None)):
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(api.app)

    def post(self, messages):
        r = self.client.post("/v1/chat/completions", json={
            "model": "deepseek-chat", "messages": messages})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["choices"][0]["message"]["content"]

    def test_the_block_follows_the_newest_message_every_turn(self):
        first = [{"role": "system", "content": self.SYSTEM},
                 {"role": "user", "content": "."},
                 {"role": "assistant", "content": "*Greeting.*"},
                 {"role": "user", "content": "sadking: *Nikki cuts her off.*"}]
        reply = self.post(first)
        prompt, cid = self.prompts[0]
        self.assertIsNone(cid)                                   # a new chat
        self.assertNotIn("[POST-HISTORY]", prompt)
        self.assertTrue(prompt.startswith("RULES. Write well."))
        self.assertIn("CARD: Nikki.", prompt)
        self.assertTrue(prompt.rstrip().endswith("first, in full."))
        self.assertLess(prompt.index("Nikki cuts her off"), prompt.index("Write out every"))
        # A reply that copies the block has the copied line removed.
        self.assertNotIn("Write out every direction", reply)

        second = first + [{"role": "assistant", "content": reply},
                          {"role": "user", "content": "sadking: *Liria leaves.*"}]
        self.post(second)
        prompt, cid = self.prompts[1]
        self.assertEqual(cid, "sess:2")                          # the thread goes on
        self.assertEqual(prompt, "sadking: *Liria leaves.*\n\nWrite out every direction "
                                 "in sadking's message first, in full.")

    def test_without_markers_nothing_moves(self):
        self.post([{"role": "system", "content": "RULES."},
                   {"role": "user", "content": "hi"}])
        self.assertEqual(self.prompts[0][0], "RULES.\n\nhi")
