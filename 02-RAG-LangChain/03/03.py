from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

graph = Neo4jGraph(
    url="bolt://localhost:7687", 
    username="neo4j", 
    password="0305Goldwing",
    database="datacampdb02-01"
)

llm = ChatOpenAI(api_key=openai_api_key, temperature=0)

graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph, 
    verbose=True,
    allow_dangerous_requests=True
) 
# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who discovered the element Radium?"})

# Print the result text
print(f"Final answer: {result['result']}")