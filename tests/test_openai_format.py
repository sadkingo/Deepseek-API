import unittest
from server.schemas import ChatMessage
from server.openai_format import (
    _canonical_text,
    _est_tokens,
    completion_response,
    messages_to_prompt,
    strip_role_leak,
    serialize_tool_call,
    extract_tool_calls,
)


class TestOpenAIFormat(unittest.TestCase):
    def test_token_estimation(self):
        text = "Hello world, this is a test prompt."
        tokens = _est_tokens(text)
        self.assertGreaterEqual(tokens, 1)

    def test_canonical_text_plain(self):
        msg = ChatMessage(role="user", content="Hello DeepSeek!")
        self.assertEqual(_canonical_text(msg), "Hello DeepSeek!")

    def test_canonical_text_developer_role(self):
        msg = ChatMessage(role="developer", content="You are a helpful assistant.")
        self.assertEqual(_canonical_text(msg), "You are a helpful assistant.")

    def test_canonical_text_tool_call(self):
        msg = ChatMessage(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        )
        canon = _canonical_text(msg)
        self.assertIn("Let me check.", canon)
        self.assertIn("<function_call>", canon)
        self.assertIn("get_weather", canon)

    def test_completion_response_shape(self):
        resp = completion_response(
            model="deepseek-chat",
            content="Hello!",
            prompt="Hi",
            conversation_id="test_session:1",
            reasoning="Thinking...",
        )
        self.assertEqual(resp["object"], "chat.completion")
        self.assertEqual(resp["model"], "deepseek-chat")
        self.assertEqual(resp["conversation_id"], "test_session:1")
        choice = resp["choices"][0]
        self.assertEqual(choice["message"]["content"], "Hello!")
        self.assertEqual(choice["message"]["reasoning_content"], "Thinking...")
        self.assertIn("usage", resp)


if __name__ == "__main__":
    unittest.main()


class ResendPrompt(unittest.TestCase):
    """A chat resumed with several messages at once (the 2026-11-01 case: the
    previous reply had been edited, so the user's last message, the edited
    reply and the new message went out together, and the model skipped the
    narration in the new message as if it had already been shown)."""

    def msgs(self, *pairs):
        from server.schemas import ChatMessage
        return [ChatMessage(role=r, content=c) for r, c in pairs]

    def test_the_newest_message_is_labelled_as_the_one_to_answer(self):
        from server.openai_format import RESEND_EARLIER, RESEND_NEWEST, resend_prompt
        p = resend_prompt(self.msgs(
            ("user", "sadking: *she thinks it over*"),
            ("assistant", "*Liria watches her.*"),
            ("user", "sadking: *Nikki cuts her off. It's not about cheating.*")))
        self.assertEqual(p, f"{RESEND_EARLIER}\n\nsadking: *she thinks it over*\n\n"
                            f"*Liria watches her.*\n\n{RESEND_NEWEST}\n\n"
                            "sadking: *Nikki cuts her off. It's not about cheating.*")

    def test_everything_else_goes_verbatim(self):
        from server.openai_format import messages_to_prompt, resend_prompt
        from server.schemas import ChatMessage
        cases = [
            self.msgs(("user", "just one")),
            self.msgs(("user", "two"), ("user", "user messages")),     # no reply among them
            self.msgs(("user", "q"), ("assistant", "a")),               # ends with a reply
            [ChatMessage(role="assistant", content="", tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "read", "arguments": "{}"}}]),
             ChatMessage(role="tool", content="file", tool_call_id="c1"),
             ChatMessage(role="user", content="next")],                 # tool traffic
        ]
        for m in cases:
            self.assertEqual(resend_prompt(m), messages_to_prompt(m))


class WholeHistoryPrompt(unittest.TestCase):
    def test_system_then_earlier_turns_then_the_newest_message(self):
        from server.openai_format import RESEND_EARLIER, RESEND_NEWEST, whole_history_prompt
        p = whole_history_prompt([
            ChatMessage(role="system", content="RULES"),
            ChatMessage(role="user", content="."),
            ChatMessage(role="assistant", content="*Greeting.*"),
            ChatMessage(role="user", content="sadking: *Nikki cuts her off.*")])
        self.assertEqual(p, f"RULES\n\n{RESEND_EARLIER}\n\n.\n\n*Greeting.*\n\n"
                            f"{RESEND_NEWEST}\n\nsadking: *Nikki cuts her off.*")

    def test_agent_histories_and_bare_messages_go_verbatim(self):
        from server.openai_format import messages_to_prompt, whole_history_prompt
        for msgs in ([ChatMessage(role="system", content="S"), ChatMessage(role="user", content="hi")],
                     [ChatMessage(role="user", content="q"), ChatMessage(role="assistant", content="a")]):
            self.assertEqual(whole_history_prompt(msgs), messages_to_prompt(msgs))
