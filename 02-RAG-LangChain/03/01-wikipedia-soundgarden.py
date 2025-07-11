from langchain_community.document_loaders import WikipediaLoader 
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_community.graphs import Neo4jGraph

openai_api_key = os.getenv("OPENAI_API_KEY")

raw_documents = WikipediaLoader(query="Soundgarden").load() 
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)  
documents = text_splitter.split_documents(raw_documents[:5])   

# Save all documents to text file
with open('wikipedia_soundgarden_output.txt', 'w', encoding='utf-8') as f:
    for i, doc in enumerate(documents):
        f.write(f"\n--- Document {i+1} ---\n")
        f.write(f"Title: {doc.metadata.get('title', 'N/A')}\n")
        f.write(f"URL: {doc.metadata.get('source', 'N/A')}\n")
        f.write(f"Content length: {len(doc.page_content)} characters\n")
        f.write("Content preview:\n")
        f.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        f.write("\n" + "-" * 50 + "\n")

print("Output saved to 'wikipedia_soundgarden_output.txt'")


llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model_name="gpt-4o-mini") 
llm_transformer = LLMGraphTransformer(llm=llm) 

graph_documents = llm_transformer.convert_to_graph_documents(documents) 
# print(graph_documents) 

url = "bolt://localhost:7687"
user = "neo4j"
password = "0305Goldwing"

graph = Neo4jGraph(url=url, username=user, password=password, database="soundgarden")

graph.add_graph_documents( 
    graph_documents,  
    include_source=True, 
    baseEntityLabel=True 
) 

graph.refresh_schema()
schema = graph.get_schema
print("Schema loaded:\n")
print(schema)






