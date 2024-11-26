from openai import OpenAI, AsyncOpenAI

from gmg_auto.NLI_node_extraction import NodeExtractor
from gmg_auto.NLI_extract_edges import EdgeExtractor
from gmg_auto.NLI_suggest_node_distribution import NodeDistributer
from gmg_auto.graph_class import GMG_graph


class NaturalLanguageInput:
    """
    some docstring
    """

    def __init__(self, openai_client: OpenAI | AsyncOpenAI) -> None:
        self.descr = None
        self.openai_client = openai_client

    def fit(self, description: str):
        """just saves the description of the graph

        Args:
            description (str): description of the graph

        Returns:
            Self: NaturalLanguageInput object
        """

        self.descr = description

        # now initialize all extractors
        self.node_exractor = NodeExtractor(self.openai_client)
        self.edge_extractor = EdgeExtractor(self.openai_client)
        self.node_distributer = NodeDistributer(self.openai_client)

    def construct_graph(
        self, gpt_model: str = "gpt-4o-mini", temperature: float = 0
    ) -> GMG_graph:
        nodes = self.node_exractor.extract_nodes_gpt(self.descr, gpt_model, temperature)
        edges = self.edge_extractor.extract_all_edges(
            self.descr, nodes, gpt_model, temperature
        )
        node_distributions = self.node_distributer.suggest_vertex_distributions(
            self.descr, nodes, gpt_model, temperature
        )

        return GMG_graph(nodes, edges, node_distributions)
