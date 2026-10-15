import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot import sanitize, get_response, RESPONSES, FALLBACK, EXIT_COMMANDS


class TestSanitize(unittest.TestCase):
    def test_case_and_whitespace(self):
        self.assertEqual(sanitize("  HeLLo  "), "hello")

    def test_trailing_punctuation(self):
        self.assertEqual(sanitize("Hello!!!"), "hello")

    def test_inner_spaces(self):
        self.assertEqual(sanitize("what   is   ai?"), "what is ai")


class TestResponses(unittest.TestCase):
    def test_known_intent(self):
        self.assertIn(get_response("hello"), RESPONSES["hello"])

    def test_fallback(self):
        self.assertEqual(get_response("asdfgh"), FALLBACK)

    def test_at_least_five_intents(self):
        self.assertGreaterEqual(len(RESPONSES), 5)

    def test_exit_commands(self):
        self.assertIn("bye", EXIT_COMMANDS)
        self.assertIn("exit", EXIT_COMMANDS)


if __name__ == "__main__":
    unittest.main()
