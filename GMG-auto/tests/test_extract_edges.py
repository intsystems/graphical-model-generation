import unittest
import os
import sys
from unittest import mock
from unittest.mock import MagicMock

# Add the code directory to the system path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import NLI_extract_edges
from NLI_extract_edges import EdgeExtractor


class TestEdgeExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.edge_extractor = EdgeExtractor(MagicMock())

    def test_get_mess_type(self):
        messs = self.edge_extractor._get_messages_for_edge_direction(
            "desr", ["a"], ("a", "a")
        )

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

    @mock.patch("NLI_extract_edges.node_extraction_sys_message", "You are assistant.")
    @mock.patch(
        "NLI_extract_edges.edge_extraction_str_template",
        "Description: {description}, set_of_nodes: {set_of_nodes}, pair_of_nodes: {pair_of_nodes}",
    )
    def test_get_mess_text(self):

        description = "my_description"
        format_dict = {
            "description": "my_description",
            "set_of_nodes": ["a", "b"],
            "pair_of_nodes": ("a", "b"),
        }

        messages = self.edge_extractor._get_messages_for_edge_direction(**format_dict)
        value = (
            {
                "role": "system",
                "content": NLI_extract_edges.node_extraction_sys_message,
            },
            {
                "role": "user",
                "content": NLI_extract_edges.edge_extraction_str_template.format(
                    **format_dict
                ),
            },
        )

        self.assertEqual(
            value,
            messages,
            f"Wrong message formation!\nmessage:{messages}\nshould be: {value}",
        )

    @mock.patch.object(EdgeExtractor, "_get_completion_parsed_result")
    @mock.patch(
        "NLI_extract_edges.EdgeExtractor._get_messages_for_edge_direction",
        MagicMock(return_value=["some messages"]),
    )
    def test_extract_one_edge_gpt(self, mock_completion):
        call_args = {
            "description": "description",
            "set_of_nodes": ["a", "b"],
            "pair_of_nodes": ("a", "b"),
            "gpt_model": "gpt-4o-mini",
            "temperature": 0.0,
        }

        # test on forward
        mock_completion.return_value = "forward"
        result = self.edge_extractor._extract_one_edge_gpt(**call_args)
        self.assertEqual(result, ("a", "b"))

        # test on backward
        mock_completion.return_value = "backward"
        result = self.edge_extractor._extract_one_edge_gpt(**call_args)
        self.assertEqual(result, ("b", "a"))

        # test on NO edge
        mock_completion.return_value = "no"
        result = self.edge_extractor._extract_one_edge_gpt(**call_args)
        self.assertEqual(result, (None, None))

        # test on inside call
        inside_call_args = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "messages": ["some messages"],
            "response_format": NLI_extract_edges.ArrowType,
            "key": "arrow_type",
        }
        mock_completion.assert_called_with(**inside_call_args)

    @mock.patch(
        "NLI_extract_edges.EdgeExtractor._extract_one_edge_gpt",
        MagicMock(return_value=("a", "a")),
    )
    @mock.patch("builtins.print", MagicMock(return_value="None"))
    def test_extract_all_edges_gpt(self):
        call_args = {
            "description": "description",
            "set_of_nodes": ["a", "b"],
            "gpt_model": "gpt-4o-mini",
            "temperature": 0.0,
            "verbose": True,
        }

        result = self.edge_extractor.extract_all_edges(**call_args)
        self.assertEqual(result, [("a", "a")])

    def test_init(self):
        mock_client = MagicMock()
        mock_node_extractor = EdgeExtractor(mock_client)

        self.assertEqual(mock_node_extractor.openai_client, mock_client)

    @classmethod
    def tearDownClass(self) -> None:
        del self.edge_extractor


if __name__ == "main":
    unittest.main()
