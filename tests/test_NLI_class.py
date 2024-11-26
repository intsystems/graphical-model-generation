import unittest
import os
import sys
from unittest import mock
from unittest.mock import MagicMock

# Add the code directory to the system path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import NLI_class
from NLI_class import NaturalLanguageInput
import openai
from openai import OpenAI


class TestNLIClass(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.mock_openAI_client = MagicMock()
        self.nli = NaturalLanguageInput(self.mock_openAI_client)

    @mock.patch("NLI_class.NodeExtractor")
    @mock.patch("NLI_class.EdgeExtractor")
    @mock.patch("NLI_class.NodeDistributer")
    def test_fit(self, node_distr, edge_extr, node_extr):

        self.nli.fit("descr")

        self.assertEqual(self.nli.descr, "descr")

        node_distr.assert_called_once_with(self.mock_openAI_client)
        edge_extr.assert_called_once_with(self.mock_openAI_client)
        node_extr.assert_called_once_with(self.mock_openAI_client)


    @mock.patch(
        "NLI_class.NodeExtractor.extract_nodes_gpt",
        MagicMock(return_value=(["a"])),
    )
    @mock.patch(
        "NLI_class.EdgeExtractor.extract_all_edges",
        MagicMock(return_value=([])),
    )
    @mock.patch.object(
        "NLI_class.NodeDistributer.suggest_vertex_distributions",
        MagicMock(return_value=(['binary'])),
    )
    @mock.patch(
        "NLI_class.GMG_graph",
        MagicMock(return_value=('graph created')),
    )
    def test_construct_graph(self, node_distr):

        res = self.nli.construct_graph()

        self.assertEqual(res, 'graph created')
        node_distr.assert_called_once_with(None, "gpt-4o-mini", 0.0)

        # and all the same

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

    @classmethod
    def tearDownClass(self) -> None:
        del self.mock_openAI_client
        del self.nli


if __name__ == "main":
    unittest.main()
