from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

openai_api_key = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("02-RAG-LangChain/AuditSummaryDraft-ENG_CMPY-157914_ACTY-2022-532608.pdf")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    ['\n', '.', ' ', ''],
    chunk_size=1000,  # Increased from 75
    chunk_overlap=100  # Increased from 10
)
# Split the document using text_splitter
chunks = text_splitter.split_documents(document)


prompt = """
Use the only the context provided to answer the following question. If you don't know the answer, reply that you are unsure.
Context: {context}
Question: {question}
"""

# Convert the string into a chat prompt template
prompt_template =  ChatPromptTemplate.from_template(prompt)


llm = ChatOpenAI( 
    api_key=openai_api_key, 
    model="gpt-4o-mini" 
)

# Create a BM25 retriever from chunks
retriever = BM25Retriever.from_documents(chunks, k=7)

# Create the LCEL retrieval chain
chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt_template
         | llm
         | StrOutputParser()
)

print(chain.invoke("Extract finding detail?"))




