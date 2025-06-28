import pandas as pd
import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language 
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import shutil
from langchain_community.document_loaders import PythonLoader


openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a document loader for README.md and load it
loader = UnstructuredMarkdownLoader("02-RAG-LangChain/README.md")

markdown_data = loader.load()
# print(markdown_data[0])

# Create a document loader for rag.py and load it
loader = PythonLoader('02-RAG-LangChain/rag.py')

python_data = loader.load()
print(python_data[0])


# Create a Python-aware recursive character splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=100
)
# Split the Python content into chunks
chunks = python_splitter.split_documents(python_data) 

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")