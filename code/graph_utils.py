import pgmpy
from pgmpy.base import DAG

def construct_graph(node_names, edge_list, node_distrs=None):
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

def visualize_graph(G: pgmpy.base.DAG, g_name='graph'):
    viz = G.to_graphviz()
    viz.draw(f'{g_name}.png', prog='neato')
    return Image(f'{g_name}.png')
