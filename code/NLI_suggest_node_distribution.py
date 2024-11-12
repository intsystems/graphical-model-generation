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

def get_messages_for_vertex_distribution(descr: str, node_name: str) -> str:

    node_extraction_sys_message = '''
            ###ROLE###
            You are Graphical model scientist.
            You task is to extract information about DAG from text description.
            The structure, or topology, of the network should capture qualitative relationships between variables. In particular, two nodes should be connected directly if one affects or causes the other, with the arc indicating the direction of the effect.
            The presence of arrows or arcs seems to imply, at an intuitive level, that for each arc one variable should be interpreted as a cause and the other as an effect (e.g., A →E means that A causes E). This interpretation is called causal.

            For every right answer I give you 5$.
        '''

    node_extraction_str_template = f'''
        ###TASK###
        You are given the graph description and name of its node.
        Depending on node name and description of graph, choose what types of values can the node take.
        If value of a node can take only 2 values, answer 'binary', if can take only limited number of discrete values, answer 'categorical',
        if its values are continious, answer 'coutinious'.

        ###OUTPUT FORMAT###
        You should answer only 1 word: 'binary', 'categorical', or 'coutinious'.

        ###
        
        GRAPH DESCRIPTION: {descr}
        NODE NAME: {node_name}
        ASSISTANT: 
    '''


    messages = [
        {
            'role': 'system',
            'content': node_extraction_sys_message
        },
        {
            'role': 'user',
            'content': node_extraction_str_template
        }
    ]

    return messages

def suggest_vertex_distribution(descr, node_name, gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    '''
        returns either (None, None) or edge with identified direction
    '''
    completion = gpt_client.chat.completions.create(
        model=gpt_model,
        messages=get_messages_for_vertex_distribution(descr, node_name),
        # response_format=ArrowType,
        temperature=temperature,
    )
    distr_type = completion.choices[0].message.content

    return distr_type

async def async_suggest_vertex_distribution(descr, node_name, async_gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    '''
        returns either (None, None) or edge with identified direction
    '''
    completion = await async_gpt_client.chat.completions.create(
        model=gpt_model,
        messages=get_messages_for_vertex_distribution(descr, node_name),
        # response_format=ArrowType,
        temperature=temperature,
    )
    distr_type = completion.choices[0].message.content

    return distr_type

async def async_suggest_vertex_distributions(descr, node_names, async_gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    node_distr_dict = {}
    for node_name in node_names:
        node_distr = await async_suggest_vertex_distribution(descr, node_name, async_gpt_client, gpt_model, temperature)
        node_distr_dict[node_name] = node_distr

    return node_distr_dict
