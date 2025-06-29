from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI 
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
import json

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key,  temperature=0)


llm_transformer = LLMGraphTransformer(llm=llm)

famous_scientists = [
    Document(
        metadata={}, 
        page_content='\nThe 20th century witnessed the rise of some of the most influential scientists in history, with Albert Einstein and Marie Curie standing out among them. Einstein, best known for his theory of relativity, revolutionized our understanding of space, time, and energy, earning him the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Marie Curie, a pioneer in the study of radioactivity, was the first woman to win a Nobel Prize. She was awarded the Nobel Prize in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel, for their work on radiation. Curie later made history again by winning a second Nobel Prize in Chemistry in 1911 for her discoveries of radium and polonium. Both scientists made monumental contributions that continue to influence the fields of physics and beyond.\n'
    )
]
graph_documents = llm_transformer.convert_to_graph_documents(famous_scientists) 
print(f"Derived Nodes:\n{graph_documents[0].nodes}\n")
print(f"Derived Edges:\n{graph_documents[0].relationships}")

def graph_to_dict(graph_doc):
    return {
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "properties": node.properties
            }
            for node in graph_doc.nodes
        ],
        "relationships": [
            {
                "source": rel.source.id,
                "target": rel.target.id,
                "type": rel.type,
                "properties": rel.properties
            }
            for rel in graph_doc.relationships
        ]
    }

# Assuming your graph_documents list has one element
graph_dict = graph_to_dict(graph_documents[0])

# Export to JSON file
with open("graph_output.json", "w", encoding="utf-8") as f:
    json.dump(graph_dict, f, indent=2)

print("Graph exported to graph_output.json ✅")