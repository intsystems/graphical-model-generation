from typing import List
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from config import node_extraction_str_template, node_extraction_sys_message
from NLI_base_extractor import BaseExtractior

class Nodes(BaseModel):
    list_of_nodes: List[str]

class NodeExtractor(BaseExtractior):
    """
        A class to handle node extraction from a given text description using OpenAI's GPT models.
    """

    def __init__(self, openai_client: OpenAI|AsyncOpenAI) -> None:
        """
            Initializes the NodeExtractor with an OpenAI client.

            Args:
                openai_client (OpenAI | AsyncOpenAI): An instance of OpenAI or AsyncOpenAI client to interact with the API.
        """
        super(NodeExtractor, self).__init__(openai_client)

    def extract_nodes_gpt(self,
                          description: str,
                          gpt_model: str = 'gpt-4o-mini', 
                          temperature: float = 0.0
    ) -> list[str]:
        """
            Extracts nodes from a given description using a specified GPT model.

            Args:
                description (str): The text description from which nodes need to be extracted.
                gpt_model (str, optional): The GPT model to use for extraction. Defaults to 'gpt-4o-mini'.
                temperature (float, optional): The randomness of the model's output. Defaults to 0.0.

            Returns:
                list[str]: A list of extracted nodes from the description.
        """        
        
        list_of_nodes = self._get_completion_parsed_result(
            model=gpt_model,
            messages=self._get_messages_for_node_extraction(description),
            response_format=Nodes,
            temperature=temperature,
            key='list_of_nodes'
        )

        return list_of_nodes
    

    @staticmethod
    def _get_messages_for_node_extraction(description: str) -> tuple[dict, dict]:
        """
            Prepares the message payload for node extraction using the GPT model.

            Args:
                description (str): The description text to be processed.

            Returns:
                tuple[dict, dict]: A tuple containing system and user messages formatted for the API request.
        """        
        messages = (
            {
                'role': 'system',
                'content': node_extraction_sys_message
            },
            {
                'role': 'user',
                'content': node_extraction_str_template.format(description=description)
            }
        )

        return messages
