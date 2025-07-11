from neo4j import GraphDatabase
import json

# === 1. Connect to your local instance ===
uri = "bolt://localhost:7687"  # Default local Bolt URI
user = "neo4j"
password = "0305Goldwing"  # <- enter the password you set in Neo4j Desktop
dbname = "datacampdb"


driver = GraphDatabase.driver(uri, auth=(user, password))
print("Connected to Neo4j ✅")


# === 2. Load JSON file ===
with open("graph_output.json", "r", encoding="utf-8") as f:
    graph_data = json.load(f)

# === 3. Define the import logic ===
def create_graph(tx, node, label):
    tx.run(f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
    """, id=node["id"], props=node["properties"])


def create_relationship(tx, rel):
    tx.run("""
        MATCH (a {id: $source})
        MATCH (b {id: $target})
        MERGE (a)-[r:%s]->(b)
        SET r += $props
    """ % rel["type"], source=rel["source"], target=rel["target"], props=rel["properties"])


# === 4. Run the transactions ===
with driver.session(database=dbname) as session:
    for node in graph_data["nodes"]:
        session.write_transaction(create_graph, node, node["type"])

    for rel in graph_data["relationships"]:
        session.write_transaction(create_relationship, rel)

print("✅ Graph imported into datacampdb!")
driver.close()
