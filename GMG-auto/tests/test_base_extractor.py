import unittest
import os
import sys
from unittest import mock
from unittest.mock import MagicMock

# Add the code directory to the system path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import NLI_base_extractor
from NLI_base_extractor import BaseExtractior
import openai
from openai import OpenAI


class TestBaseExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.mock_openAI_client = MagicMock()
        self.base_extractor = BaseExtractior(self.mock_openAI_client)

    def test_get_compl_result(self):

        self.mock_openAI_client.chat.completions.create.return_value = MagicMock()
        self.base_extractor._get_completion_result(
            "gpt-4o-mini", [{"role": "system", "message": "message"}], 0
        )

        self.mock_openAI_client.chat.completions.create.assert_called_once()

    def test_get_compl_parced_result(self):

        message_mock = MagicMock()
        message_mock.message.content = """{"key": "value"}"""
        mocked_return_value = MagicMock()
        mocked_return_value.choices = [message_mock]

        self.mock_openAI_client.beta.chat.completions.parse.return_value = (
            mocked_return_value
        )
        self.base_extractor._get_completion_parsed_result(
            "gpt-4o-mini", [{"role": "system", "message": "message"}], list, 0, "key"
        )

        self.mock_openAI_client.beta.chat.completions.parse.assert_called_once()

    @classmethod
    def tearDownClass(self) -> None:
        del self.base_extractor


if __name__ == "main":
    unittest.main()
