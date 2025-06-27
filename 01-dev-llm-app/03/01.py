import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader


# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader("01-dev-llm-app/03/rag_vs_fine_tuning.pdf")

# Load the document
data = loader.load() 

# Explore the loaded data
print(f"Number of pages: {len(data)}")
print(f"First page content: {data[0].page_content[:200]}...")  # First 200 chars
print(f"First page metadata: {data[0].metadata}")

# If you want to see all pages
for i, doc in enumerate(data):
    print(f"Page {i+1} has {len(doc.page_content)} characters")