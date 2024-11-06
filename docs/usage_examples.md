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