# Natural Language Graph Model Extraction Library

## Introduction

Welcome to the Natural Language Graph Model Extraction Library. This library allows users to extract and analyze edges and vertex distributions in graphical models using natural language descriptions. It leverages OpenAI's GPT models to interpret these descriptions and determine relationships and properties within the model.

## Features

- Extract directed edges between nodes using natural language descriptions.
- Suggest vertex distributions based on node names and graph descriptions.
- Supports asynchronous operations for efficient processing.

## Installation

To install the library, you can use pip:

```bash
pip install GMG_auto
```


Ensure you have access to the OpenAI GPT models required by this library.

## Usage

### Node Extraction

To extract all nodes of a graph, given a description:
```python
from GMG_auto import extract_nodes_gpt
from openai import OpenAI

gpt_client = OpenAI(
    base_url='https://api.proxyapi.ru/openai/v1', # if needed
    api_key='OPENAI_API_KEY'
)

description = '''Think about a classroom where student learning is shaped by different factors. The amount of Time Spent Studying directly influences Knowledge Acquisition. Teacher Quality also affects how well students understand the material. Classroom Environment, such as noise levels and seating arrangements, can impact both Teacher Quality and Knowledge Acquisition. Altogether, these elements contribute to a student's overall Learning Outcome.'''.lower()

extracted_nodes = extract_nodes_gpt(description, gpt_client, gpt_model='gpt-4o-mini', temperature=0)
print(extracted_nodes)
> ['Time Spent Studying', 'Knowledge Acquisition', 'Teacher Quality', 'Classroom Environment', 'Learning Outcome']
```

### Edge Extraction

#### Extracting a Single Edge

To extract a single edge direction between two nodes:
```python
from GMG_auto import extract_one_edge_gpt
from openai import OpenAI

gpt_client = OpenAI(
    base_url='https://api.proxyapi.ru/openai/v1', # if needed
    api_key='OPENAI_API_KEY'
)

description = "Think about a classroom where student learning is shaped by different factors. The amount of Time Spent Studying directly influences Knowledge Acquisition. Teacher Quality also affects how well students understand the material. Classroom Environment, such as noise levels and seating arrangements, can impact both Teacher Quality and Knowledge Acquisition. Altogether, these elements contribute to a student's overall Learning Outcome"
all_nodes = ['Time Spent Studying', 'Knowledge Acquisition', 'Teacher Quality', 'Classroom Environment', 'Learning Outcome']
node1 = "Learning Outcome"
node2 = "Teacher Quality"

edge = extract_one_edge_gpt(description, all_nodes, (node1, node2), gpt_client, gpt_model='gpt-4o-mini', temperature=0)

print(edge)
> ('Teacher Quality', 'Learning Outcome')
```


#### Extracting All Edges

To extract all possible edges in a set of nodes:
```python
from GMG_auto import extract_all_edges
from openai import AsyncOpenAI

async_gpt_client = AsyncOpenAI(
    base_url='https://api.proxyapi.ru/openai/v1', # if needed
    api_key='OPENAI_API_KEY'
)

description = "Think about a classroom where student learning is shaped by different factors. The amount of Time Spent Studying directly influences Knowledge Acquisition. Teacher Quality also affects how well students understand the material. Classroom Environment, such as noise levels and seating arrangements, can impact both Teacher Quality and Knowledge Acquisition. Altogether, these elements contribute to a student's overall Learning Outcome"
all_nodes = ['Time Spent Studying', 'Knowledge Acquisition', 'Teacher Quality', 'Classroom Environment', 'Learning Outcome']

edges = await extract_all_edges(description, setofnodes, async_gpt_client, gpt_model='gpt-4o-mini', temperature=0)
print("Extracted Edges:")
for edge in edges:
    print(edge)
```

### Vertex Distribution Suggestion

To suggest a vertex distribution for a node:

```python
from GMG_auto import suggest_vertex_distribution
from openai import OpenAI

gpt_client = OpenAI(
    base_url='https://api.proxyapi.ru/openai/v1', # if needed
    api_key='OPENAI_API_KEY'
)

description = "Think about a classroom where student learning is shaped by different factors. The amount of Time Spent Studying directly influences Knowledge Acquisition. Teacher Quality also affects how well students understand the material. Classroom Environment, such as noise levels and seating arrangements, can impact both Teacher Quality and Knowledge Acquisition. Altogether, these elements contribute to a student's overall Learning Outcome"
all_nodes = ['Time Spent Studying', 'Knowledge Acquisition', 'Teacher Quality', 'Classroom Environment', 'Learning Outcome']
node_name = "Teacher Quality"

distribution = suggest_vertex_distribution(description, node_name, gpt_client, gpt_model='gpt-4o-mini', temperature=0)
print(f"distribution")
> Categorical
```


## API Reference

### Functions

#### extract_one_edge_gpt(description: str, node1: str, node2: str) -> Tuple[Optional[str], Optional[str]]
- **Description**: Extracts the direction of an edge between two nodes based on a description.
- **Parameters**:
  - description: A string describing the graph.
  - node1: The first node.
  - node2: The second node.
- **Returns**: A tuple representing the directed edge or (None, None) if no direction is found.

#### async_extract_one_edge_gpt(description: str, node1: str, node2: str) -> Awaitable[Tuple[Optional[str], Optional[str]]]
- **Description**: Asynchronously extracts the direction of an edge between two nodes.

#### extract_all_edges(description: str, set_of_nodes: Set[str]) -> List[Tuple[str, str]]
- **Description**: Extracts all possible directed edges from a set of nodes based on a description.

#### suggest_vertex_distribution(node_name: str, graph_description: str) -> str
- **Description**: Suggests the type of distribution for a node based on its name and graph description.

## Asynchronous Operations

The library supports asynchronous operations to improve efficiency when processing multiple requests. Ensure your environment supports asynchronous programming if you choose to use these features.
