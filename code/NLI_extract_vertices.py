from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from enum import Enum
import os
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate
from openai import OpenAI, AsyncOpenAI
from typing import List
import json

class ArrowEnum(str, Enum):
    no = "no arrow"
    forward = "forward arrow"
    backward = "backward arrow"

class ArrowType(BaseModel):
    arrow_type: ArrowEnum

def get_messages_for_edge_direction(descr: str, set_of_nodes: list[str], pair_of_nodes: tuple[str]) -> str:

    node_extraction_sys_message = '''
            ###ROLE###
            You are Graphical model scientist.
            You task is to extract information about DAG from text description.
            The structure, or topology, of the network should capture qualitative relationships between variables. In particular, two nodes should be connected directly if one affects or causes the other, with the arc indicating the direction of the effect.
            The presence of arrows or arcs seems to imply, at an intuitive level, that for each arc one variable should be interpreted as a cause and the other as an effect (e.g., A →E means that A causes E). This interpretation is called causal.

            For every right answer I give you 5$.
        '''

    edge_extraction_str_template = f'''
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
            #DESCRIPTION#: {descr}
            #SET OF NODES#: {set_of_nodes}
            #PAIR OF NODES#: {pair_of_nodes}

        Assistant:
    '''


    messages = [
        {
            'role': 'system',
            'content': node_extraction_sys_message
        },
        {
            'role': 'user',
            'content': edge_extraction_str_template
        }
    ]

    return messages

def extract_one_edge_gpt(descr, set_of_nodes, pair_of_nodes, gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    '''
        returns either (None, None) or edge with identified direction
    '''
    completion = gpt_client.beta.chat.completions.parse(
        model=gpt_model,
        messages=get_messages_for_edge_direction(descr, f"[{', '.join(set_of_nodes)}]", f"[{', '.join(pair_of_nodes)}]"),
        response_format=ArrowType,
        temperature=temperature,
    )
    arrow_type = json.loads(completion.choices[0].message.content)['arrow_type']  # json.loads(completion.choices[0].message.content)['list_of_nodes']

    if 'forward' in arrow_type.lower():
        return pair_of_nodes
    if 'backward' in arrow_type.lower():
        return pair_of_nodes[::-1]
    return (None, None)

# AsyncOpenAI
async def async_extract_one_edge_gpt(descr, set_of_nodes, pair_of_nodes, async_gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    '''
        returns either (None, None) or edge with identified direction
    '''
    completion = await async_gpt_client.beta.chat.completions.parse(
        model=gpt_model,
        messages=get_messages_for_edge_direction(descr, f"[{', '.join(set_of_nodes)}]", f"[{', '.join(pair_of_nodes)}]"),
        response_format=ArrowType,
        temperature=temperature,
    )
    arrow_type = json.loads(completion.choices[0].message.content)['arrow_type']  # json.loads(completion.choices[0].message.content)['list_of_nodes']

    if 'forward' in arrow_type.lower():
        return pair_of_nodes
    if 'backward' in arrow_type.lower():
        return pair_of_nodes[::-1]
    return (None, None)

async def extract_all_edges(descr, set_of_nodes, async_gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    edge_list = []

    for i, node_a in enumerate(set_of_nodes):
        for node_b in set_of_nodes[i+1:]:
            print(f"{node_a} # {node_b}")
            edge = await async_extract_one_edge_gpt(descr, set_of_nodes, (node_a, node_b), async_gpt_client, gpt_model=gpt_model, temperature=temperature)
            if edge[0] is not None:
                edge_list.append(edge)

    return edge_list
