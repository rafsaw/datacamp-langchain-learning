import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Load environment variables from .env file
load_dotenv()

url = os.getenv("VECTOR_DB_URL")
user = os.getenv("VECTOR_DB_USER")
password = os.getenv("VECTOR_DB_PASSWORD")
db = os.getenv("VECTOR_DB_NAME")

graph = Neo4jGraph(url=url, username=user, password=password, database=db)

query = "MATCH (b:Band {id: 'Soundgarden'})-[:RELEASED]->(a:Album) RETURN a"
results = graph.query(query)
print(results)






# schema = graph.get_schema
# print("Schema loaded:\n")
# print(schema)






