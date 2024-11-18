# Natural Language Graph Model Extraction Library

## Introduction

Welcome to the Natural Language Graph Model Extraction Library. This library allows users to extract and analyze edges and vertex distributions in graphical models using natural language descriptions. It leverages OpenAI's GPT models to interpret these descriptions and determine relationships and properties within the model.

## Features

- Extract directed edges between nodes using natural language descriptions.
- Suggest vertex distributions based on node names and graph descriptions.
- Supports asynchronous operations for efficient processing.


## Project layout (TO DO)

    README.md             # About the repo.
    mkdocs.yml            # The configuration file.
    GMG_LLM_powered.pdf   # Presentation of a project.

    docs/
        index.md  # The documentation homepage.
        installation.md
        usage_examplex.md
        function_descriptions.md

    code/
        config.py
        __init__.py
        NLI_base_extractor.py
        NLI_node_extractor.py
        NLI_edge_extractor.py
        NLI_suggest_node_distribution.py
        graph_utils.py

    tests/
        test_base_extractor.py
        test_extract_nodes.py
        test_extract_edges.py
        test_suggest_node_distr.py

    
