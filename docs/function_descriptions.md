# API Reference

## Functions

### Graph utils
::: code.graph_utils

### Node extraction
::: code.NLI_node_extraction.extract_nodes_gpt

### Node distribution inference
::: code.NLI_suggest_vertex_distribution

#### Edge extraction
::: code.NLI_extract_vertices



#### extract_one_edge_gpt(description: str, node1: str, node2: str) -> Tuple[Optional[str], Optional[str]]
- **Description**: Extracts the direction of an edge between two nodes based on a description.
- **Parameters**:
  - description: A string describing the graph.
  - node1: The first node.
  - node2: The second node.
- **Returns**: A tuple representing the directed edge or (None, None) if no direction is found.

::: code.graph_utils
<!-- ::: GMG.NLI_extract_verticies.get_messages_for_edge_direction -->
<!-- .extract_one_edge_gpt -->


## Asynchronous Operations

The library supports asynchronous operations to improve efficiency when processing multiple requests. Ensure your environment supports asynchronous programming if you choose to use these features.
