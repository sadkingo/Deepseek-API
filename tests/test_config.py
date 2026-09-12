import unittest
from server.config import (
    MODEL_MAP,
    is_known_model,
    resolve_model_type,
    model_thinking,
    resolve_alias,
)


class TestConfig(unittest.TestCase):
    def test_known_models(self):
        self.assertTrue(is_known_model("deepseek-chat"))
        self.assertTrue(is_known_model("deepseek-reasoner"))
        self.assertTrue(is_known_model("deepseek-expert"))
        self.assertTrue(is_known_model("deepseek-expert-reasoner"))
        self.assertFalse(is_known_model("nonexistent-model"))

    def test_resolve_model_type(self):
        self.assertEqual(resolve_model_type("deepseek-chat"), "default")
        self.assertEqual(resolve_model_type("deepseek-reasoner"), "default")
        self.assertEqual(resolve_model_type("deepseek-expert"), "expert")
        self.assertEqual(resolve_model_type("deepseek-expert-reasoner"), "expert")

    def test_model_thinking(self):
        self.assertFalse(model_thinking("deepseek-chat"))
        self.assertTrue(model_thinking("deepseek-reasoner"))
        self.assertFalse(model_thinking("deepseek-expert"))
        self.assertTrue(model_thinking("deepseek-expert-reasoner"))

    def test_resolve_alias_passthrough(self):
        self.assertEqual(resolve_alias("deepseek-chat"), "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
