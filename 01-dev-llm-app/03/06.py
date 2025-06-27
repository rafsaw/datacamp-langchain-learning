import pandas as pd
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the HTML document into memory
loader = UnstructuredHTMLLoader("01-dev-llm-app/03/Further Extending the TikTok Enforcement Delay – The White House.html") 
data = loader.load()

# Define variables
chunk_size = 300
chunk_overlap = 100

# Split the HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=['.'])

docs = splitter.split_documents(data)

# Analyze the results
print(f"Original document length: {len(data[0].page_content)} characters")
print(f"Number of chunks created: {len(docs)}")
print(f"Average chunk size: {sum(len(doc.page_content) for doc in docs) / len(docs):.1f} characters")

# Show first few chunks
print("\nFirst 3 chunks:")
for i, doc in enumerate(docs[:3]):
    print(f"\nChunk {i+1} ({len(doc.page_content)} chars):")
    print(f"'{doc.page_content[:100]}...'")
    print(f"Metadata: {doc.metadata}")