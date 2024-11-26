from NLI_class import NaturalLanguageInput
from openai import OpenAI

client = OpenAI(
    base_url='https://api.proxyapi.ru/openai/v1',
    api_key="sk-3IqehGd2A8iAldf9VgG2CgHyGQtO9qPH",
)

nli = NaturalLanguageInput(client)

example = '''Think about a classroom where student learning is shaped by different factors. The amount of Time Spent Studying directly influences Knowledge Acquisition. Teacher Quality also affects how well students understand the material. Classroom Environment, such as noise levels and seating arrangements, can impact both Teacher Quality and Knowledge Acquisition. Altogether, these elements contribute to a student's overall Learning Outcome.'''.lower()

nli.fit(example)

print("Fitted")

G = nli.construct_graph()

print('G constructed')

print(G.nodes)
print(G.edges)
print(G.node_distrs)