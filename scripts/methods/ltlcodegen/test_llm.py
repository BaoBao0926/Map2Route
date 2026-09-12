import unittest
import urllib.error
from unittest import mock

from scripts.methods.ltlcodegen import llm


class LTLCodeGenLLMTest(unittest.TestCase):
    def test_translate_retries_transient_url_error(self) -> None:
        inventory = {
            "rooms": ["room_1: kitchen"],
            "objects": [],
        }

        response_patch = mock.patch.object(
            llm,
            "gemini_response_text",
            side_effect=[
                urllib.error.URLError("temporary name resolution failure"),
                'def question():\n    return ap("room_1")',
            ],
        )
        sleep_patch = mock.patch.object(llm.time, "sleep")
        with response_patch as response_text:
            with sleep_patch as sleep:
                translation = llm.translate_instruction_to_ltl(
                    instruction="go to the kitchen",
                    inventory=inventory,
                    model="test-model",
                    max_retries=2,
                )

        self.assertEqual(response_text.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(translation["ltl_formula"], "room_1")
        self.assertEqual(
            translation["attempts"],
            [
                {
                    "attempt": "1",
                    "status": "failed",
                    "failure_reason": "<urlopen error temporary name resolution failure>",
                },
                {"attempt": "2", "status": "success"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
