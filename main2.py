import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load your OpenAI key from .env
load_dotenv()

# Set up the model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a prompt
prompt = PromptTemplate.from_template("You are an assistant. Answer the following question:\n\nQuestion: {question}")

# Create chain using new syntax
chain = prompt | llm

# Run the chain
response = chain.invoke({"question": "What is LangChain (the Python framework) and why is it useful?"})
print(response.content)
