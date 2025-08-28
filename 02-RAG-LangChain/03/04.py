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



# Your current working setup
graph1 = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="0305Goldwing", database="datacampdb02-01")


# Compare results
chain1 = GraphCypherQAChain.from_llm(llm=llm, graph=graph1, verbose=True, allow_dangerous_requests=True)
chain2 = GraphCypherQAChain.from_llm(llm=llm, graph=graph1, verbose=True, allow_dangerous_requests=True, exclude_types=["Concept"])

result1 = chain1.invoke({"query": "Who discovered the element Radium?"})
result2 = chain2.invoke({"query": "Who discovered the element Radium?"})

print(result1)
print(result2)