import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent 
from langchain_community.agent_toolkits.load_tools import load_tools 

##ReAct agent

# Load your OpenAI key from .env
load_dotenv()

# Create an OpenAI chat LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Define the tools
tools = load_tools(["wikipedia"])

# Define the agent
agent = create_react_agent(llm, tools)

# Invoke the agent
response = agent.invoke({"messages": [("human", "How many people live in New York City?")]})
# print(response)
print(response['messages'][-1].content) 