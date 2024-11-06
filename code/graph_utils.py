import pgmpy
from pgmpy.base import DAG
from typing import List, Tuple
from IPython.display import Image

def construct_graph(node_names: List[str], edge_list: List[Tuple[str, str]], node_distrs=None) -> DAG:
    '''
        constructs graph from nodes and lists
    '''
    G = DAG()

    # add nodes
    try:
        G.add_nodes_from(node_names)
    except:
        print('Something wrong with nodex')
        raise

    
    # add edges
    try:
        G.add_edges_from(ebunch=edge_list)
    except:
        print('Something wrong with nodex')
        raise

    return G

def visualize_graph(G: pgmpy.base.DAG, g_name='graph') -> Image:
    '''
        Visualizes G, saves into .png file
    '''
    viz = G.to_graphviz()
    viz.draw(f'{g_name}.png', prog='neato')
    return Image(f'{g_name}.png')
