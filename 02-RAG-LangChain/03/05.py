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

# Alternative with excluded concepts  
graph2 = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="0305Goldwing", database="datacampdb02-01")

# Compare results with validation
chain1 = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph1, 
    verbose=True, 
    allow_dangerous_requests=True,
    validate_cypher=True  # ✅ Validate Cypher syntax
)

chain2 = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph2, 
    verbose=True, 
    allow_dangerous_requests=True, 
    exclude_types=["Concept"],
    validate_cypher=True  # ✅ Validate Cypher syntax
)

result1 = chain1.invoke({"query": "Who won the Nobel Prize In Physics?"})
result2 = chain2.invoke({"query": "Who won the Nobel Prize In Physics?"})

print("Without exclude_types:")
print(result1)
print("\nWith exclude_types=['Concept']:")
print(result2)