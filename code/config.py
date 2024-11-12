'''Contains all text constants, used in the application.
'''

### Node extraction

node_extraction_sys_message = '''
    ###ROLE###
    You are Graphical model scientist.
    You task is to extract information about DAG from text description.
    The structure, or topology, of the network should capture qualitative relationships between variables. In particular, two nodes should be connected directly if one affects or causes the other, with the arc indicating the direction of the effect.
    The presence of arrows or arcs seems to imply, at an intuitive level, that for each arc one variable should be interpreted as a cause and the other as an effect (e.g., A →E means that A causes E). This interpretation is called causal.

    For every right answer I give you 5$.
'''

node_extraction_str_template = '''
    ###TASK###
    Extract all node names of GAD from the description, the user gaves you.

    ###OUTPUT FORMAT###
    You output should be list of strings

    ###

    User: Imagine a garden where the growth of plants depends on several factors. The amount of Watering affects how well the plants grow. Sunlight is another crucial factor, as it provides energy for photosynthesis. Fertilizer also plays a role by supplying essential nutrients. Together, these factors influence Plant Growth. Additionally, Watering can impact the effectiveness of Fertilizer, as nutrients are better absorbed when the soil is moist.
    Assistant: ['Watering', 'Sunlight', 'Fertilizer', 'Plant Growth']

    ###

    User: {description}
    Assistant:
'''

### Edge extraction

edge_extraction_str_template = '''
    ###TASK###
    You are given a DAG description, a set of its nodes and pair of nodes.
    You should infer from the description and you own knowledge the type of casuality between two given nodes:
        - forward: the left may be the cause of the right
        - backward: the right may be the cause of the left
        - no: no direct casualities

    ###OUTPUT FORMAT###
    You output should be "forward", "backward" or "no".

    ###

    User:
        #DESCRIPTION#: Imagine a garden where the growth of plants depends on several factors. The amount of Watering affects how well the plants grow. Sunlight is another crucial factor, as it provides energy for photosynthesis. Fertilizer also plays a role by supplying essential nutrients. Together, these factors influence Plant Growth. Additionally, Watering can impact the effectiveness of Fertilizer, as nutrients are better absorbed when the soil is moist.
        #SET OF NODES#: [Watering, Sunlight, Fertilizer, Plant Growth]
        #PAIR OF NODES#: (Sunlight, Plant Growth)

    Assistant: forward

    ###

    User:
        #DESCRIPTION#: Imagine a garden where the growth of plants depends on several factors. The amount of Watering affects how well the plants grow. Sunlight is another crucial factor, as it provides energy for photosynthesis. Fertilizer also plays a role by supplying essential nutrients. Together, these factors influence Plant Growth. Additionally, Watering can impact the effectiveness of Fertilizer, as nutrients are better absorbed when the soil is moist.
        #SET OF NODES#: [Watering, Sunlight, Fertilizer, Plant Growth]
        #PAIR OF NODES#: (Fertilizer, Watering)

    Assistant: backward

    ###

    User:
        #DESCRIPTION#: Imagine a garden where the growth of plants depends on several factors. The amount of Watering affects how well the plants grow. Sunlight is another crucial factor, as it provides energy for photosynthesis. Fertilizer also plays a role by supplying essential nutrients. Together, these factors influence Plant Growth. Additionally, Watering can impact the effectiveness of Fertilizer, as nutrients are better absorbed when the soil is moist.
        #SET OF NODES#: [Watering, Sunlight, Fertilizer, Plant Growth]
        #PAIR OF NODES#: (Sunlight, Fertilizer)

    Assistant: no

    ###

    User:
        #DESCRIPTION#: {description}
        #SET OF NODES#: {set_of_nodes}
        #PAIR OF NODES#: {pair_of_nodes}

    Assistant:
'''

### Edge direction
node_distribution_str_template = '''
    ###TASK###
    You are given the graph description and name of its node.
    Depending on node name and description of graph, choose what types of values can the node take.
    If value of a node can take only 2 values, answer 'binary', if can take only limited number of discrete values, answer 'categorical',
    if its values are continious, answer 'continuous'.

    ###OUTPUT FORMAT###
    You should answer only 1 word: 'binary', 'categorical', or 'continuous'.

    ###
    
    GRAPH DESCRIPTION: {description}
    NODE NAME: {node_name}
    ASSISTANT: 
'''