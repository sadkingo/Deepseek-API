import unittest
from deepseek.pow import DeepSeekPow


class TestDeepSeekPow(unittest.TestCase):
    def test_pow_init_and_clean_exit(self):
        solver = DeepSeekPow()
        self.assertIsNotNone(solver._memory)
        self.assertIsNotNone(solver._solve)
        self.assertIsNotNone(solver._malloc)
        self.assertIsNotNone(solver._free)


if __name__ == "__main__":
    unittest.main()
