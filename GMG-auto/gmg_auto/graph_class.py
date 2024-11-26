import pgmpy
from pgmpy.base import DAG
from IPython.display import Image


class GMG_graph:
    def __init__(
        self, nodes: list[str], edges: list[tuple[str, str]], node_distrs: list[str]
    ) -> None:

        self.nodes = nodes
        self.edges = edges
        self.node_distrs = node_distrs

        self.G = DAG()

        # add nodes
        self.G.add_nodes_from(self.nodes)

        # add edges
        self.G.add_edges_from(ebunch=self.edges)

    def visualize(self, graph_name: str = None) -> Image:
        viz = self.G.to_graphviz()
        viz.draw(f"{graph_name}.png", prog="neato")
        return Image(f"{graph_name}.png")
