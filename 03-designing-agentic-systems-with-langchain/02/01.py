#Building graphs for chatbots 
# Building graph and agent states

import os
import math 
# Modules for structuring text 
from typing import Annotated 
from typing_extensions import TypedDict 
# LangGraph modules for defining graphs 
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages 

# Module for setting up OpenAI 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# LLM Setup 
model = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

# Define the State 
# to store the agent's data
class State(TypedDict): 
    # Define messages with metadata
    messages: Annotated[list, add_messages] 

# Define chatbot function to respond 
# with the model 
def chatbot(state: State): 
    # invoke the model with the state's messages
    return {"messages": [model.invoke(state["messages"])]}

# Initialize StateGraph 
# object to manage the agent's workflow.
graph_builder = StateGraph(State) 

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the start and end of the  
# conversation flow 
graph_builder.add_edge(START, "chatbot") 
graph_builder.add_edge("chatbot", END) 

# Compile the graph to prepare for  
# execution 
graph = graph_builder.compile()

# Define a function to execute the chatbot based on user input
def stream_graph_updates(user_input: str):
    
    # Start streaming events from the graph with the user's input
    for event in graph.stream({"messages": [("user", user_input)]}):
        
        # Retrieve and print the chatbot node responses
        for item in event.values():
            print("Agent:", item["messages"])

# Define the user query and run the chatbot
user_query = "Who is Soundgarden?"
stream_graph_updates(user_query)

 # Import modules for chatbot diagram 
# from IPython.display import  Image, display 

# Try generating and displaying  
# the graph diagram 
# try: 
#     display(Image(graph.get_graph() 
#     .draw_mermaid_png())) 
#     # Return an exception if necessary   
# except Exception: 
#     print("Additional dependencies required.") 

# try:
#     img = graph.get_graph().draw_mermaid_png()
#     with open("graph_visualization.png", "wb") as f:
#         f.write(img)
#     print("Visualization saved as 'graph_visualization.png'")
# except Exception as e:
#     print(f"Error generating visualization: {e}")

# try:
#     # Try different possible methods
#     if hasattr(graph, 'get_graph'):
#         g = graph.get_graph()
#         print("Graph has get_graph() method")
        
#         # Try to find mermaid-related methods
#         for method_name in dir(g):
#             if 'mermaid' in method_name.lower():
#                 print(f"Found method: {method_name}")
#     else:
#         print("Graph doesn't have get_graph() method")
        
#         # Try to find mermaid-related methods directly on graph
#         for method_name in dir(graph):
#             if 'mermaid' in method_name.lower():
#                 print(f"Found method: {method_name}")
                
# except Exception as e:
#     print(f"Error exploring graph methods: {e}")


# try:
#     g = graph.get_graph()
    
#     # Since we found draw_mermaid_png method
#     png_data = g.draw_mermaid_png()
    
#     # Save the PNG data to a file
#     with open("graph_visualization.png", "wb") as f:
#         f.write(png_data)
#     print("Visualization saved as 'graph_visualization.png'")
    
#     # Since we also found draw_mermaid method, let's try that too
#     # This might give us text that we can print in the terminal
#     mermaid_text = g.draw_mermaid()
#     print("\nMermaid diagram syntax:")
#     print(mermaid_text)
#     print("\nYou can copy this syntax to https://mermaid.live/ to view it online")
    
# except Exception as e:
#     print(f"Error generating visualization: {e}")