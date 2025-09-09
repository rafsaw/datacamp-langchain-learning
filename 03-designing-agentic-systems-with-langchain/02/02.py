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

 # Modules for building a Wikipedia tool 
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain_community.tools import WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Define the State 
# to store the agent's data
class State(TypedDict): 
    # Define messages with metadata
    messages: Annotated[list, add_messages] 

# Modify chatbot function to respond with Wikipedia
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



openai_api_key = os.getenv("OPENAI_API_KEY")
# LLM Setup 
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)


# Initialize Wikipedia API wrapper to fetch top 1 result 
api_wrapper = WikipediaAPIWrapper(top_k_results=1) 

# Create a Wikipedia query tool using the API wrapper 
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper) 
tools = [wikipedia_tool] 

# Bind the Wikipedia tool to the language model
llm_with_tools = llm.bind_tools(tools)

# Initialize StateGraph 
# object to manage the agent's workflow.
graph_builder = StateGraph(State) 

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Create a ToolNode to handle tool calls and add it to the graph
tool_node = ToolNode(tools=[wikipedia_tool])
graph_builder.add_node("tools", tool_node)

# Set up a condition to direct from chatbot to tool or END node
graph_builder.add_conditional_edges("chatbot", tools_condition)


# Connect tools back to chatbot and connect START and END nodes
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Modify the graph with memory checkpointing
memory = MemorySaver()  
graph = graph_builder.compile(checkpointer=memory)

# try:
#     img = graph.get_graph().draw_mermaid_png()
#     with open("graph_visualization.png", "wb") as f:
#         f.write(img)
#     # print("Visualization saved as 'graph_visualization.png'")
# except Exception as e:
#     print(f"Error generating visualization: {e}")


# Set up a streaming function for a single user
def stream_memory_responses(user_input: str):
    config = {"configurable": {"thread_id": "single_session_memory"}}
    
    # Stream the events in the graph
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        
        # Return the agent's last response
        for value in event.values():
            if "messages" in value and value["messages"]:
                last_message = value["messages"][-1]
                print( last_message.content)
                # print("--------------------------------")
                # print("Agent:", value["messages"])
                print("--------------------------------")

stream_memory_responses("Tell me about the Eiffel Tower.")
stream_memory_responses("Who built it?")
# stream_memory_responses("What is the height of the Eiffel Tower?")