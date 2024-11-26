"""
This module contains the NodeDistributer class, which is designed to suggest distribution types 
for nodes based on textual descriptions using OpenAI's GPT models. It leverages natural language 
processing to infer whether a node's distribution should be binary, categorical, or continuous.
"""

from openai import AsyncOpenAI, OpenAI
from config import node_extraction_sys_message, node_distribution_str_template
from NLI_base_extractor import BaseExtractior


class NodeDistributer(BaseExtractior):
    """
    A class for distributing nodes based on text descriptions using OpenAI's GPT models.
    """

    def __init__(self, openai_client: OpenAI | AsyncOpenAI) -> None:
        """
        Initializes the NodeDistributer with an OpenAI client.

        Args:
            openai_client (OpenAI | AsyncOpenAI): An instance of OpenAI or AsyncOpenAI client to interact with the API.
        """

        super(NodeDistributer, self).__init__(openai_client)

    @staticmethod
    def _get_messages_for_vertex_distribution(
        description: str, node_name: str
    ) -> tuple[dict, dict]:
        """
        Prepares the message payload for node extraction using the GPT model.

        Args:
            description (str): The description text to be processed.
            node_name (str): Node name for which the distribution in inferred.

        Returns:
            tuple[dict, dict]: A tuple containing system and user messages formatted for the API request.
        """
        messages = (
            {"role": "system", "content": node_extraction_sys_message},
            {
                "role": "user",
                "content": node_distribution_str_template.format(
                    description=description, node_name=node_name
                ),
            },
        )

        return messages

    def _suggest_vertex_distribution(
        self,
        description: str,
        node_name: str,
        gpt_model: str = "gpt-4o-mini",
        temperature: float = 0,
    ) -> str:
        """
        Suggests the distribution type of a node based on its description.

        Args:
            description (str): The text description of the node.
            node_name (str): The name of the node for which to suggest a distribution.
            gpt_model (str, optional): The GPT model to use for suggestion. Defaults to 'gpt-4o-mini'.
            temperature (float, optional): The randomness of the model's output. Defaults to 0.

        Returns:
            str: The suggested distribution type, which can be 'binary', 'categorical', or 'continuous'.
        """
        distr_type = self._get_completion_result(
            model=gpt_model,
            messages=self._get_messages_for_vertex_distribution(description, node_name),
            temperature=temperature,
        )

        assert distr_type in [
            "binary",
            "categorical",
            "continuous",
        ], f"Wrong inferred distr type: {distr_type}"

        return distr_type

    def suggest_vertex_distributions(
        self,
        description: str,
        node_names: list[str],
        gpt_model: str = "gpt-4o-mini",
        temperature: float = 0,
    ) -> str:
        """
        Suggests distribution types for multiple nodes based on their descriptions.

        Args:
            description (str): The text description from which to infer node distributions.
            node_names (list[str]): A list of node names for which to suggest distributions.
            gpt_model (str, optional): The GPT model to use for suggestions. Defaults to 'gpt-4o-mini'.
            temperature (float, optional): The randomness of the model's output. Defaults to 0.

        Returns:
            dict[str, str]: A dictionary mapping each node name to its suggested distribution type.
        """
        node_distr_dict = {}
        for node_name in node_names:
            node_distr = self._suggest_vertex_distribution(
                description, node_name, gpt_model, temperature
            )
            node_distr_dict[node_name] = node_distr

        return node_distr_dict
