from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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

# ✅ Fix: Escape curly braces in Cypher queries using double braces
examples = [
    {
        'question': 'How many scientists are mentioned in the graph?', 
        'query': 'MATCH (p:Person) RETURN count(DISTINCT p)'
    },
    {
        'question': 'Who won the Nobel Prize In Physics?',
        'query': 'MATCH (p:Person)-[:AWARDED]->(a:Award {{id: "Nobel Prize In Physics"}}) RETURN p.id'
    },
    {
        'question': 'What elements did Marie Curie discover?',
        'query': 'MATCH (p:Person {{id: "Marie Curie"}})-[:DISCOVERED]->(e:Element) RETURN e.id'
    },
    {
        'question': 'What is Einstein known for?',
        'query': 'MATCH (p:Person {{id: "Albert Einstein"}})-[:KNOWN_FOR]->(c:Concept) RETURN c.id'
    }
]

# Create an example prompt template
example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)

# Create the few-shot prompt template
cypher_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question", "schema"]
)

# Create the graph Cypher QA chain
graph_qa_chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm, 
    cypher_prompt=cypher_prompt,
    verbose=True, 
    validate_cypher=True,
    allow_dangerous_requests=True
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who won the Nobel Prize In Physics?"})
print(f"Final answer: {result['result']}")