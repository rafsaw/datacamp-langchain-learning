from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_document import Node, Relationship


graph_documents= [GraphDocument(nodes=[Node(id='Albert Einstein', type='Person', properties={}), Node(id='Marie Curie', type='Person', properties={}), Node(id='Nobel Prize In Physics', type='Award', properties={}), Node(id='Nobel Prize', type='Award', properties={}), Node(id='Theory Of Relativity', type='Concept', properties={}), Node(id='Photoelectric Effect', type='Concept', properties={}), Node(id='Radioactivity', type='Concept', properties={})], relationships=[Relationship(source=Node(id='Albert Einstein', type='Person', properties={}), target=Node(id='Theory Of Relativity', type='Concept', properties={}), type='KNOWN_FOR', properties={}), Relationship(source=Node(id='Albert Einstein', type='Person', properties={}), target=Node(id='Nobel Prize In Physics', type='Award', properties={}), type='AWARDED', properties={}), Relationship(source=Node(id='Albert Einstein', type='Person', properties={}), target=Node(id='Photoelectric Effect', type='Concept', properties={}), type='EXPLANATION_FOR', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Nobel Prize', type='Award', properties={}), type='AWARDED', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Radioactivity', type='Concept', properties={}), type='KNOWN_FOR', properties={})], source=Document(metadata={'source': 'scientists.txt'}, page_content='The 20th century witnessed the rise of some of the most influential scientists in history, with Albert Einstein and Marie Curie standing out among them. Einstein, best known for his theory of relativity, revolutionized our understanding of space, time, and energy, earning him the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Marie Curie, a pioneer in the study of radioactivity, was the first woman to win a Nobel Prize. She was awarded the Nobel Prize in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel, for their work on radiation. Curie later made history again by winning a second Nobel Prize in Chemistry in 1911 for her discoveries of radium and polonium. Both scientists made monumental contributions that continue to influence the fields of physics and beyond.')), GraphDocument(nodes=[Node(id='Marie Curie', type='Person', properties={}), Node(id='Nobel Prize In Physics', type='Award', properties={}), Node(id='Pierre Curie', type='Person', properties={}), Node(id='Henri Becquerel', type='Person', properties={}), Node(id='Nobel Prize In Chemistry', type='Award', properties={}), Node(id='Radium', type='Element', properties={}), Node(id='Polonium', type='Element', properties={})], relationships=[Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Nobel Prize In Physics', type='Award', properties={}), type='AWARDED', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Pierre Curie', type='Person', properties={}), type='SPOUSE', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Henri Becquerel', type='Person', properties={}), type='COLLABORATOR', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Nobel Prize In Chemistry', type='Award', properties={}), type='AWARDED', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Radium', type='Element', properties={}), type='DISCOVERED', properties={}), Relationship(source=Node(id='Marie Curie', type='Person', properties={}), target=Node(id='Polonium', type='Element', properties={}), type='DISCOVERED', properties={})], source=Document(metadata={'source': 'scientists.txt'}, page_content='The 20th century witnessed the rise of some of the most influential scientists in history, with Albert Einstein and Marie Curie standing out among them. Einstein, best known for his theory of relativity, revolutionized our understanding of space, time, and energy, earning him the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Marie Curie, a pioneer in the study of radioactivity, was the first woman to win a Nobel Prize. She was awarded the Nobel Prize in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel, for their work on radiation. Curie later made history again by winning a second Nobel Prize in Chemistry in 1911 for her discoveries of radium and polonium. Both scientists made monumental contributions that continue to influence the fields of physics and beyond.'))]


from langchain_community.graphs import Neo4jGraph

url = "bolt://localhost:7687"
user = "neo4j"
password = "0305Goldwing"

graph = Neo4jGraph(url=url, username=user, password=password, database="datacampdb02")

graph.add_graph_documents(
    graph_documents,
    include_source=True, # link nodes to source documents with MENTIONS edge
    baseEntityLabel=True # use the label of the node as the entity label
)

graph.refresh_schema()
schema = graph.get_schema
# print("Schema loaded:\n")
# print(schema)

# Query the graph
results = graph.query("""
MATCH (relativity:Concept {id: "Theory Of Relativity"}) <-[:KNOWN_FOR]- (scientist:Person)
RETURN scientist
""")

print(results[0])