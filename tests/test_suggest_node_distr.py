import unittest
import os
import sys
from unittest import mock
from unittest.mock import MagicMock

# Add the code directory to the system path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))

import NLI_suggest_node_distribution
from NLI_suggest_node_distribution import NodeDistributer


class TestNodeDistributer(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.node_distributer = NodeDistributer(MagicMock())

    def test_get_mess_type(self):
        messs = self.node_distributer._get_messages_for_vertex_distribution(
            'desr',
            'node_name'
        )
        
        # check number of messages
        self.assertEqual(len(messs), 2)

        # check type of messages
        self.assertIsInstance(messs, tuple)
        self.assertIsInstance(messs[0], dict)
        self.assertIsInstance(messs[1], dict)

        # check keys of messages
        self.assertEqual(set(messs[0].keys()), set(('role', 'content')), "wrong keys in messages")

    @mock.patch('NLI_suggest_node_distribution.node_extraction_sys_message', "You are assistant.")
    @mock.patch('NLI_suggest_node_distribution.node_distribution_str_template', "Description: {description}, node_name: {node_name}")
    def test_get_mess_text(self):

        format_dict = {
            'description': 'my_description',
            'node_name': 'node_name'
        }
        
        messages = self.node_distributer._get_messages_for_vertex_distribution(**format_dict)
        value = (
            {'role': 'system', 'content': NLI_suggest_node_distribution.node_extraction_sys_message},
            {'role': 'user', 'content': NLI_suggest_node_distribution.node_distribution_str_template.format(**format_dict)}
        )


        self.assertEqual(value, messages, f"Wrong message formation!\nmessage:{messages}\nshould be: {value}")

    
    @mock.patch('NLI_suggest_node_distribution.NodeDistributer._get_completion_result', MagicMock(return_value='binary'))
    def test_suggest_vertex_distribution(self):
        call_args = {
            'description': 'description',
            'node_name': 'a',
            'gpt_model': 'gpt-4o-mini',
            'temperature': 0.0
        }

        result = self.node_distributer._suggest_vertex_distribution(**call_args)
        self.assertEqual(result, 'binary')


    @mock.patch('NLI_suggest_node_distribution.NodeDistributer._suggest_vertex_distribution', MagicMock(return_value='binary'))
    def test_suggest_vertex_distributions(self):
        call_args = {
            'description': 'description',
            'node_names': ['a'],
            'gpt_model': 'gpt-4o-mini',
            'temperature': 0.0
        }

        result = self.node_distributer.suggest_vertex_distributions(**call_args)
        self.assertEqual(result, {'a': 'binary'})
                

    def test_init(self):
        mock_client = MagicMock()
        mock_node_distributer = NodeDistributer(mock_client)

        self.assertEqual(mock_node_distributer.openai_client, mock_client)

    @classmethod
    def tearDownClass(self) -> None:
        del self.node_distributer

if __name__ == 'main':
    unittest.main()