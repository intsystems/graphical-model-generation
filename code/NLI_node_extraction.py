
import os
from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate
from openai import OpenAI, AsyncOpenAI
from typing import List
from pydantic import BaseModel
import json

def get_messages_for_node_extraction(descr: str) -> str:

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
        Extract all node names of GAD from the description, the user gaves you.

        ###OUTPUT FORMAT###
        You output should be list of strings

        ###

        User: Imagine a garden where the growth of plants depends on several factors. The amount of Watering affects how well the plants grow. Sunlight is another crucial factor, as it provides energy for photosynthesis. Fertilizer also plays a role by supplying essential nutrients. Together, these factors influence Plant Growth. Additionally, Watering can impact the effectiveness of Fertilizer, as nutrients are better absorbed when the soil is moist.
        Assistant: ['Watering', 'Sunlight', 'Fertilizer', 'Plant Growth']

        ###

        User: {descr}
        Assistant:
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

class Nodes(BaseModel):
    list_of_nodes: List[str]

def extract_nodes_gpt(descr, gpt_client, gpt_model='gpt-4o-mini', temperature=0):
    completion = gpt_client.beta.chat.completions.parse(
        model=gpt_model,
        messages=get_messages_for_node_extraction(descr),
        response_format=Nodes,
        temperature=temperature,
    )
    return json.loads(completion.choices[0].message.content)['list_of_nodes']
