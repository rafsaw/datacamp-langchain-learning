import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load your OpenAI key from .env
load_dotenv()

# Define the LLM
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini")

# Predict the words following the text in question
prompt = 'Tell me difference between llama Index and LangChain'
response = llm.invoke(prompt)

print(response.content)