# GMG_auto 
![Coverage Badge](https://intsystems.github.io/graphical-model-generation/coverage/coverage-badge.svg)
[<img src="https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white">](https://intsystems.github.io/graphical-model-generation/)

## Motivation
The construction of graphical models is a fundamental task in probabilistic modeling and Bayesian networks. Traditionally, building these models requires expert knowledge to define variables, their relationships, and conditional dependencies. This process is often manual, time-consuming, and prone to human error. With the advent of Large Language Models (LLMs), there is an opportunity to automate this process by translating natural language descriptions directly into graphical models.


## Example of a graphical model
![image](https://github.com/user-attachments/assets/3f969879-d342-4543-90fa-847c72d8f593)

## Algo scheme
![NLP](https://github.com/user-attachments/assets/d337f04b-0e15-4891-a40c-8325fd347ebf)


## Code structure
```
├── README.md # About the repo.
├── mkdocs.yml # The configuration file.
├── GMG_LLM_powered.pdf # Presentation of a project.
│
├── code/ # Source code files
│ ├── config.py # Configuration settings.
│ ├── init.py # Package initialization.
│ ├── NLI_base_extractor.py # Base extractor for NLI.
│ ├── NLI_node_extractor.py # Node extractor for NLI.
│ ├── NLI_edge_extractor.py # Edge extractor for NLI.
│ ├── NLI_suggest_node_distribution.py # Suggests node distribution for NLI.
│ └── graph_utils.py # Utility functions for graph processing.
|
├── tests/ # Test files
│ ├── test_base_extractor.py # Tests for base extractor functionality.
│ ├── test_extract_nodes.py # Tests for node extraction functionality.
│ ├── test_extract_edges.py # Tests for edge extraction functionality.
│ └── test_suggest_node_distr.py # Tests for node distribution suggestions.
|
├── docs/ # Documentation files
│ ├── index.md # The documentation homepage.
│ ├── installation.md # Installation instructions.
│ ├── usage_examples.md # Usage examples.
│ └── function_descriptions.md # Descriptions of functions.
│
└── utils/ # Utility scripts and dependencies
├── requirements.txt # Project dependencies.
└── badge_generator.py # Generates coverage button.
```


## Algorithms to implement
**NaturalLanguageInput Class** will have the following algorithms:
1. extract_vertices
   - by the text in English, describing the probability structure, it extracts vertices, using NER or Natural Language Understanding (with LLMs and prompts)
   - it also identifies, which type of vertex is it: random variable or constant
2. extract_vertex_dependencies
   - by the text and 2 vertex names suggests type of dependency (no OR ordered link)
3. construct_graph
   - given verticies and links constructs directed graph
   - graph can save any information about verticies (name/type/etc) and links
4. correct_graph
   - given constructed directed graph, it corrects its structure (loops/unused vertives and etc.)
   - returns set of graphs, made from original with corrections
5. suggest_vertex_distributions
   - given the graph and text suggests type of distribution for every vertex
  
**GraphModel Class** will have the following algorithms:
1. visualize
   - plots graph image, with captions for verticies and specitication of distribution of every verticies
2. create_graph
   - given verticies and links creates graph
3. change_vertex_info
   - given number of vertex, changes its properties (distribution/name/etc.)

**Deploying into HF/Gradio**
- TODO

## Reccomended stack
1. **LLM API**: GPT-4o, o1, Claude 3.5, Gemini
2. **Code generation**: LLaMa2, LLaMa3
3. **LLM tuning**: Stable LM 2, Mixtral8B, Alpaca
4. **Graphs**: NetworkX
5. **Visualization**: Graphvis, Matplotlib, Plotly
6. **Deploy**: HF Spaces, Gradio


## Documentation and test coverage

Documentation available at [docs](https://intsystems.github.io/graphical-model-generation/)

Documentation and test coverage badges can be updated automatically using [github actions](.github/workflows).

Initially both of these workflows are disabled (but can be run via "Actions" page).

To enable them automatically on push to master branch, change corresponding "yaml" files.

## Contributors
- [Ernest Nasyrov](https://github.com/2001092236)
## Useful Links
- [Graph Generation with VariationalRNN 2019](https://arxiv.org/pdf/1910.01743) 
- [GraphVAE: 2018](https://arxiv.org/abs/1802.03480)
- 
