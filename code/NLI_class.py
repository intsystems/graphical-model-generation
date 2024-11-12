from typing import Self
from openai import OpenAI, AsyncOpenAI

from NLI_node_extraction import NodeExtraction

class NaturalLanguageInput:
    """
        some docstring
    """

    def __init__(self, openai_client: OpenAI|AsyncOpenAI) -> None:
        self.descr = None
        self.openai_client = openai_client
        


    def fit(self, description: str) -> Self:
        """ just saves the description of the graph

        Args:
            description (str): description of the graph

        Returns:
            Self: NaturalLanguageInput object
        """

        self.descr = description

        # now initialize all extractors
        self.node_exractor = NodeExtraction(self.openai_client)
        