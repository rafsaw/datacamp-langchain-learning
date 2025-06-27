import pandas as pd
import os
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create the customers DataFrame
customers = pd.DataFrame({
    'id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'name': [
        'Tech Innovators Inc.',
        'Green Solutions Ltd.',
        'Global Enterprises',
        'Peak Performance Co.',
        'Visionary Ventures',
        'NextGen Technologies',
        'Dynamic Dynamics LLC',
        'Infinity Services',
        'Eco-Friendly Products',
        'Future Insights'
    ],
    'subscription_type': [
        'Premium', 'Standard', 'Basic', 'Premium', 'Standard', 
        'Basic', 'Premium', 'Standard', 'Basic', 'Premium'
    ],
    'active_users': [450, 300, 150, 800, 600, 200, 700, 500, 100, 900],
    'auto_renewal': [True, False, True, True, False, True, True, False, True, True]
})

# Define a function to retrieve customer info by-name
def retrieve_customer_info_step1(name: str) -> str:
    """Retrieve customer information based on their name."""
    # Filter customers for the customer's name
    customer_info = customers[customers['name'] == name]
    return customer_info.to_string()

# Define a function to retrieve customer info by-name
@tool
def retrieve_customer_info(name: str) -> str:
    """Retrieve customer information based on their name."""
    # Filter customers for the customer's name
    customer_info = customers[customers['name'] == name]
    return customer_info.to_string()


  
# print(retrieve_customer_info.name)          # "financial_report"
# print(retrieve_customer_info.description)   # The docstring
# print(retrieve_customer_info.return_direct) # False
# print(retrieve_customer_info.args)          # Dictionary of argument types

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)
# Create agent with custom tool
agent = create_react_agent(llm, [retrieve_customer_info])

# Invoke the agent on the input
messages = agent.invoke({"messages": [("human", "Create a summary of our customer: Peak Performance Co.")]})

# Print the result
print(messages['messages'][-1].content)