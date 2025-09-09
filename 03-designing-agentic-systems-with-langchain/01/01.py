#DESIGNING AGENTIC SYSTEMS WITH LANGCHAIN

#Create a ReAct agent

import os
from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI 
from langgraph.prebuilt import create_react_agent 
import math 
from dotenv import load_dotenv

load_dotenv()

# Define the missing count_r_in_word tool
@tool
def count_r_in_word(word: str) -> int:
    """
    Counts how many lowercase 'r's are in a given word.
    
    Args:
        word: The word to count 'r's in.
        
    Returns:
        The number of 'r's in the word.
    """
    return word.lower().count('r')


openai_api_key = os.getenv("OPENAI_API_KEY")
# LLM Setup 
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

# Create a ReAct agent
 # Create the agent 
agent = create_react_agent(model, tools=[count_r_in_word]) 

# Create a query
query = "How many r's are in the word 'Terrarium'?"

# Invoke the agent and print the response 
response = agent.invoke({"messages": [("human", query)]})

# Print the agent's response 
print(response['messages'][-1].content) 