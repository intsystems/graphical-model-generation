import unittest
import os
import sys
from unittest import mock
from unittest.mock import MagicMock

# Add the code directory to the system path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import NLI_node_extraction
from NLI_node_extraction import NodeExtractor


class TestNodeExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.node_extractor = NodeExtractor(MagicMock())

    def test_get_mess_type(self):
        messs = self.node_extractor._get_messages_for_node_extraction("desr")

        # check number of messages
        self.assertEqual(len(messs), 2)

        # check type of messages
        self.assertIsInstance(messs, tuple)
        self.assertIsInstance(messs[0], dict)
        self.assertIsInstance(messs[1], dict)

        # check keys of messages
        self.assertEqual(
            set(messs[0].keys()), set(("role", "content")), "wrong keys in messages"
        )

    @mock.patch("NLI_node_extraction.node_extraction_sys_message", "You are assistant.")
    @mock.patch(
        "NLI_node_extraction.node_extraction_str_template",
        "Extract nodes: {description}",
    )
    def test_get_mess_text(self):

        description = "my_description"
        messages = self.node_extractor._get_messages_for_node_extraction(description)
        value = (
            {
                "role": "system",
                "content": NLI_node_extraction.node_extraction_sys_message,
            },
            {
                "role": "user",
                "content": NLI_node_extraction.node_extraction_str_template.format(
                    description=description
                ),
            },
        )

        self.assertEqual(
            value,
            messages,
            f"Wrong message formation!\nmessage:{messages}\nshould be: {value}",
        )

    @mock.patch.object(
        NodeExtractor, "_get_completion_parsed_result", return_value=["a", "b", "c"]
    )
    @mock.patch.object(
        NodeExtractor,
        "_get_messages_for_node_extraction",
        return_value=["some messages"],
    )
    @mock.patch("NLI_node_extraction.Nodes", list)
    def test_extract_nodes_gpt(self, mock_get_messages, mock_completion):
        call_args = {
            "description": "description",
            "gpt_model": "gpt-4o-mini",
            "temperature": 0.0,
        }

        # obj = NodeExtractor("no")
        result = self.node_extractor.extract_nodes_gpt(**call_args)

        self.assertEqual(result, ["a", "b", "c"])

        inside_call_args = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "messages": ["some messages"],
            "response_format": list,
            "key": "list_of_nodes",
        }
        mock_completion.assert_called_once_with(**inside_call_args)

    def test_init(self):
        mock_client = MagicMock()
        mock_node_extractor = NodeExtractor(mock_client)

        self.assertEqual(mock_node_extractor.openai_client, mock_client)

    @classmethod
    def tearDownClass(self) -> None:
        del self.node_extractor


if __name__ == "main":
    unittest.main()
