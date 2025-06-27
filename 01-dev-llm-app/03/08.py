import pandas as pd
import os

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# Add placeholders to the message string
message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])


# # Test the prompt template
# sample_context = "RAG (Retrieval-Augmented Generation) combines the power of retrieval systems with large language models. It allows models to access external knowledge during generation."
# sample_question = "What is RAG?"

# # Format the prompt
# formatted_prompt = prompt_template.invoke({
#     "context": sample_context,
#     "question": sample_question
# })

# print("Formatted prompt:")
# print(formatted_prompt.to_string())


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Embed the documents in a persistent Chroma vector database
embedding_function = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-small')

# Load the database
db = Chroma(persist_directory="rag_vs_fine_tuning_db", embedding_function=embedding_function)

retriever = db.as_retriever( 
    search_type="similarity", 
    search_kwargs={"k": 3} 
) 

# Create LLM
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")

# retriever | format_docs means:
# Grab response from vector database (retriever finds relevant document chunks)
# Convert into string (format_docs function joins them into a single text)

# Query: "What is RAG?"
#     ↓
# retriever: Search vector DB → [Doc1, Doc2, Doc3]
#     ↓  
# format_docs: Convert to string → "RAG is...\n\nRAG allows...\n\nRAG combines..."
#     ↓
# prompt_template: Insert into {context} placeholder
#     ↓
# LLM: Generate answer based on context


# Create the complete RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

# Test the complete RAG system
response = rag_chain.invoke("What are the main advantages of RAG over fine-tuning?")
print(response.content)

response1 = rag_chain.invoke("Extract all the technical requirements mentioned for fine-tuning")

response2 = rag_chain.invoke("What specific benefits does RAG provide for real-time information?")

response3 = rag_chain.invoke("List only the costs mentioned in the document")

response4 = rag_chain.invoke("What does the document say about training time?")

#put all the responses into a dataframe and save it to a csv file
df = pd.DataFrame({
    "Question": [
        "Extract all the technical requirements mentioned for fine-tuning",
        "What specific benefits does RAG provide for real-time information?",
        "List only the costs mentioned in the document",
        "What does the document say about training time?"
    ],
    "Response": [
        response1.content,
        response2.content,
        response3.content,
        response4.content
    ]
})

df.to_csv("rag_vs_fine_tuning_responses.csv", index=False)
    