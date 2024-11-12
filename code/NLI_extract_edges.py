"""Code for vertices extraction.

This module provides functionality to extract edges between nodes in a graph using OpenAI's GPT models. It includes methods to determine the existence and direction of edges based on a textual description.

The module contains the following functions:

- _extract_one_edge_gpt: Determines the existence and direction of an edge between a pair of nodes.
- extract_all_edges: Extracts all possible edges from a set of nodes based on a given description.
"""


from pydantic import BaseModel
from enum import Enum
from openai import OpenAI, AsyncOpenAI
from NLI_base_extractor import BaseExtractior
from config import node_extraction_sys_message, edge_extraction_str_template


class ArrowEnum(str, Enum):
    """
        Enumeration for arrow direction types in a graph.
    """
    no = "no arrow"
    forward = "forward arrow"
    backward = "backward arrow"


class ArrowType(BaseModel):
    """
        Model representing the type of arrow direction for an edge.
    """
    arrow_type: ArrowEnum


class EdgeExtractor(BaseExtractior):
    """
        Class to handle edge extraction from a description using OpenAI's GPT models.
    """

    def __init__(self, openai_client: OpenAI|AsyncOpenAI) -> None:
        """
            Initializes the NodeExtraction with an OpenAI client.

            Args:
                openai_client (OpenAI | AsyncOpenAI): An instance of OpenAI or AsyncOpenAI client to interact with the API.
        """
        super(EdgeExtractor, self).__init__(openai_client)

    
    @staticmethod
    def _get_messages_for_edge_direction(description: str, 
                                         set_of_nodes: list[str], 
                                         pair_of_nodes: tuple[str, str],
    ) -> tuple[dict, dict]:
        """
            Prepares the message payload for edge direction identification using the GPT model.

            Args:
                description (str): The description text to be processed.
                set_of_nodes (list[str]): The nodes of the graph.
                pair_of_nodes (tuple[str, str]): Pair of nodes to decide about the existance and the direction of the edge.

            Returns:
                tuple[dict, dict]: A tuple containing system and user messages formatted for the API request.
        """  

        messages = [
            {
                'role': 'system',
                'content': node_extraction_sys_message
            },
            {
                'role': 'user',
                'content': edge_extraction_str_template.format(description=description, set_of_nodes=set_of_nodes, pair_of_nodes=pair_of_nodes)
            }
        ]

        return messages


    def _extract_one_edge_gpt(self,
                             description: str, 
                             set_of_nodes: list[str],
                             pair_of_nodes: tuple[str, str], 
                             gpt_model: str = 'gpt-4o-mini',
                             temperature: float = 0,
    ) -> tuple[str|None, str|None]:
        """
            Determines the existence and direction of an edge between a pair of nodes using a GPT model.

            Args:
                description (str): The description text to be processed.
                set_of_nodes (list[str]): The nodes of the graph.
                pair_of_nodes (tuple[str, str]): A pair of nodes to check for an edge.
                gpt_model (str, optional): The GPT model to use for extraction. Defaults to 'gpt-4o-mini'.
                temperature (float, optional): The randomness of the model's output. Defaults to 0.

            Returns:
                tuple[str | None, str | None]: either (None, None) if no edge exists, or a tuple representing the edge with identified direction.
        """

        arrow_type = self._get_completion_parsed_result(
            model=gpt_model,
            messages=self._get_messages_for_edge_direction(description, f"[{', '.join(set_of_nodes)}]", f"[{', '.join(pair_of_nodes)}]"),
            response_format=ArrowType,
            temperature=temperature,
            key='arrow_type'
        )

        if 'forward' in arrow_type.lower():
            return pair_of_nodes
        if 'backward' in arrow_type.lower():
            return pair_of_nodes[::-1]
        
        return (None, None)
    

    def extract_all_edges(self,
                                description: str, 
                                set_of_nodes: list[str],
                                gpt_model: str = 'gpt-4o-mini',
                                temperature: float = 0,
                                verbose=False
    ) -> tuple[str|None, str|None]:
        """
            Extracts all possible edges from a set of nodes based on a given description.

            Args:
                description (str): The description text to be processed.
                set_of_nodes (list[str]): The nodes of the graph.
                gpt_model (str, optional): The GPT model to use for extraction. Defaults to 'gpt-4o-mini'.
                temperature (float, optional): The randomness of the model's output. Defaults to 0.
                verbose (bool, optional): If True, provides detailed logging. Defaults to False.

            Returns:
                list[tuple[str | None, str | None]]: A list of tuples representing edges with identified directions.
        """    
        edge_list = []

        for i, node_a in enumerate(set_of_nodes):
            for node_b in set_of_nodes[i+1:]:
                if verbose:
                    print(f"{node_a} # {node_b}")
                edge = self._extract_one_edge_gpt(description, set_of_nodes, (node_a, node_b), gpt_model, temperature)
                if edge[0] is not None:
                    edge_list.append(edge)

        return edge_list