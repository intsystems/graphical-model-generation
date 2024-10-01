# GMG_auto 

[<img src="coverage-badge.svg">](https://github.com/intsystems/SoftwareTemplate-simplified/tree/master)
[<img src="https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white">](https://intsystems.github.io/SoftwareTemplate-simplified)

## Motivation
The construction of graphical models is a fundamental task in probabilistic modeling and Bayesian networks. Traditionally, building these models requires expert knowledge to define variables, their relationships, and conditional dependencies. This process is often manual, time-consuming, and prone to human error. With the advent of Large Language Models (LLMs), there is an opportunity to automate this process by translating natural language descriptions directly into graphical models.

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
Documentation and test coverage badges can be updated automatically using [github actions](.github/workflows).

Initially both of these workflows are disabled (but can be run via "Actions" page).

To enable them automatically on push to master branch, change corresponding "yaml" files.

## Contributors
- [Ernest Nasyrov](https://github.com/2001092236)
## Useful Links
- [Graph Generation with VariationalRNN 2019](https://arxiv.org/pdf/1910.01743) 
- [GraphVAE: 2018](https://arxiv.org/abs/1802.03480)
- 
