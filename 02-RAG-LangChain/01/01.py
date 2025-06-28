import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader("02-RAG-LangChain/2005.11401v4.pdf")

# Load the document
data = loader.load() 

print(data[0])

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader('02-RAG-LangChain/datacamp-blog.html')

# Load the document
data = loader.load()

# Print the first document's content
print(data[0].page_content)

# Print the first document's metadata
print(data[0].metadata)