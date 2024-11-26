import unittest
import os
import sys
from unittest import mock
from unittest.mock import MagicMock

# Add the code directory to the system path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import graph_class
from graph_class import GMG_graph


class TestGraphClass(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.nodes = ["a", "b", "c"]
        self.edges = [
            ("a", "b"),
            ("b", "c"),
            ("a", "c"),
        ]

        self.node_distrs = ["categorical", "binary", "continious"]

        self.G = GMG_graph(self.nodes, self.edges, self.node_distrs)

    def test_init(self):
        self.assertEqual(self.G.nodes, self.nodes)
        self.assertEqual(self.G.edges, self.edges)
        self.assertEqual(self.G.node_distrs, self.node_distrs)

    @mock.patch(
        "Ipython.display.Image",
        MagicMock(return_value=["mess"]),
    )
    def test_visualize(self):

        res = self.G.visualize("name")
        self.assertEqual(res, "mess")

        # message_mock = MagicMock()
        # message_mock.message.content = """{"key": "value"}"""
        # mocked_return_value = MagicMock()
        # mocked_return_value.choices = [message_mock]

        # self.mock_openAI_client.beta.chat.completions.parse.return_value = (
        #     mocked_return_value
        # )
        # self.base_extractor._get_completion_parsed_result(
        #     "gpt-4o-mini", [{"role": "system", "message": "message"}], list, 0, "key"
        # )

        # self.mock_openAI_client.beta.chat.completions.parse.assert_called_once()


if __name__ == "main":
    unittest.main()
