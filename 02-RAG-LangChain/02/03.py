from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader





chunks = [
    "RAG stands for Retrieval Augmented Generation.",
    "Graph Retrieval Augmented Generation uses graphs to store and utilize relationships between documents in the retrieval process.",
    "There are different types of RAG architectures; for example, Graph RAG."
]

# Initialize the BM25 retriever
bm25_retriever = BM25Retriever.from_texts(chunks, k=2)

# Invoke the retriever
results = bm25_retriever.invoke("Graph RAG")

# Extract the page content from the first result
print("Most Relevant Document:")
print(results[0].page_content)



