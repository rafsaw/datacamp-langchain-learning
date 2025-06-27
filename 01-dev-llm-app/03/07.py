import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader("01-dev-llm-app/03/rag_vs_fine_tuning.pdf")

# Load the document
data = loader.load() 

# Split the document using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""])
docs = splitter.split_documents(data) 


# Embed the documents in a persistent Chroma vector database
embedding_function = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-small')

# Create a Chroma vector database
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="rag_vs_fine_tuning_db")

# Save the database
vectorstore.persist()

# Load the database
db = Chroma(persist_directory="rag_vs_fine_tuning_db", embedding_function=embedding_function)

retriever = vectorstore.as_retriever( 
search_type="similarity", 
search_kwargs={"k": 3} 
) 

# Test the retriever
query = "What are the main differences between RAG and fine-tuning?"
relevant_docs = retriever.invoke(query)

print(f"Found {len(relevant_docs)} relevant chunks:")
for i, doc in enumerate(relevant_docs):
    print(f"\nChunk {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")