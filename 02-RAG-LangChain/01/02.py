import pandas as pd
import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import shutil

openai_api_key = os.getenv("OPENAI_API_KEY")

text = '''RAG (retrieval augmented generation) is an advanced NLP model that combines retrieval mechanisms with generative capabilities. RAG aims to improve the accuracy and relevance of its outputs by grounding responses in precise, contextually appropriate data.'''

# Define a text splitter that splits on the '.' character
text_splitter = CharacterTextSplitter( 
separator=".", 
chunk_size=75,   
chunk_overlap=10   
) 

# Split the text using text_splitter
chunks = text_splitter.split_text(text)
# print(chunks)
# print([len(chunk) for chunk in chunks])

loader = PyPDFLoader("02-RAG-LangChain/2005.11401v4.pdf")
document = loader.load()

# Define a text splitter that splits recursively through the character list
text_splitter = RecursiveCharacterTextSplitter(
    ['\n', '.', ' ', ''],
    chunk_size=500,  # Increased from 75
    chunk_overlap=50  # Increased from 10
)

# Split the document using text_splitter
chunks = text_splitter.split_documents(document)
# print(chunks)
# print([len(chunk.page_content) for chunk in chunks])

# Remove existing vector store to recreate with new chunking
if os.path.exists("02-RAG-LangChain/rag_store_02"):
    shutil.rmtree("02-RAG-LangChain/rag_store_02")

embedding_model = OpenAIEmbeddings( 
    api_key=openai_api_key, 
    model="text-embedding-3-small" 
) 

vector_store = Chroma.from_documents( 
    documents=chunks,  
    embedding=embedding_model,
    persist_directory="02-RAG-LangChain/rag_store_02"
)

llm = ChatOpenAI( 
    api_key=openai_api_key, 
    model="gpt-4o-mini" 
)

prompt = """
Use the only the context provided to answer the following question. If you don't know the answer, reply that you are unsure.
Context: {context}
Question: {question}
"""

# Convert the string into a chat prompt template
prompt_template =  ChatPromptTemplate.from_template(prompt)

# Create an LCEL chain to test the prompt
# chain = prompt_template | llm

# Invoke the chain on the inputs provided
# print(chain.invoke({"context": "DataCamp's RAG course was created by Meri Nova and James Chapman!", "question": "Who created DataCamp's RAG course?"}))

# Convert the vector store into a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Debug: Let's see what the retriever finds
# print("=== DEBUG: What the retriever finds ===")
# docs = retriever.get_relevant_documents("Who are the authors?")
# for i, doc in enumerate(docs):
#     print(f"Document {i+1}:")
#     print(f"Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print("---")

# Create the LCEL retrieval chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser() 
)

# Invoke the chain
print(chain.invoke("Who are the authors?"))