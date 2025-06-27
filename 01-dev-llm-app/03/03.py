import pandas as pd
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader("01-dev-llm-app/03/Further Extending the TikTok Enforcement Delay – The White House.html") 

# Load the document
data = loader.load()

# Analyze the loaded content
print(f"Number of documents created: {len(data)}")
print(f"Content length: {len(data[0].page_content)} characters")
print(f"First 300 characters:\n{data[0].page_content[:300]}...")
print(f"\nMetadata: {data[0].metadata}")

# Look for specific content patterns
content = data[0].page_content
if "TikTok" in content:
    print(f"\n'TikTok' appears {content.count('TikTok')} times in the document")