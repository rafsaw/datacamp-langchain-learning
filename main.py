from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# Optional: load environment variable from .env
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
question = "What is LangChain and why is it useful?"
template = "You are an assistant. Answer the following question:\n\nQuestion: {question}"
prompt = PromptTemplate.from_template(template)

# Use the modern RunnableSequence pattern
chain = prompt | llm

response = chain.invoke({"question": "What is LangChain and why is it useful?"})
print(response)
