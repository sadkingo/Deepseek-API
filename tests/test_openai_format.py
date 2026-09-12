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
