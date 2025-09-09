import os
import math 

from typing import Annotated 
from typing_extensions import TypedDict 

from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages 

from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain_community.tools import WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
load_dotenv()

@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the length of the hypotenuse of a right-angled triangle given the lengths of the other two sides."""
    
    # Split the input string to get the lengths of the triangle
    sides = input.split(',')
    
    # Convert the input values to floats, removing extra spaces
    a = float(sides[0].strip())
    b = float(sides[1].strip())
    
    # Square each of the values, add them together, and find the square root 
    return math.sqrt(a**2 + b**2)

openai_api_key = os.getenv("OPENAI_API_KEY")
llmOpenAI = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

api_wrapper = WikipediaAPIWrapper(top_k_results=1) 
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper) 
my_tools = [wikipedia_tool, hypotenuse_length] 

llm_with_tools = llmOpenAI.bind_tools(my_tools)



class State(TypedDict): 
    messages: Annotated[list, add_messages]

# Define the chatbot function with debug printing
def chatbot(state: State):
    # Print the state to see its structure
    print("\n=== CHATBOT NODE RECEIVED STATE ===")
    print(f"Type of state: {type(state)}")
    print(f"State keys: {state.keys()}")
    print(f"Messages in state: {state['messages']}")
    print("=== END STATE PRINTOUT ===\n")
    
    # Continue with normal function operation
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Create a wrapper function for the tool node that shows tool execution
def tool_node_wrapper(state: State):
    print("\n=== TOOL NODE EXECUTING TOOLS ===")
    
    # Get the tool calls from the last message
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for i, tool_call in enumerate(last_message.tool_calls, 1):
            print(f"Executing tool {i}: {tool_call['name']}")
            print(f"  with arguments: {tool_call['args']}")
    
    # Execute the tools and get the result
    result = tool_node.invoke(state)
    
    print("Tools execution complete")
    print("=== END TOOL EXECUTION ===\n")
    
    return result


graph_builder = StateGraph(State) 
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=my_tools)
# graph_builder.add_node("tools", tool_node)
# Add the wrapper instead of the direct tool node
graph_builder.add_node("tools", tool_node_wrapper)

graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def stream_tool_responses(user_input: "str"):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for event_name, _ in event.items():
            print("event:", event_name)
            print("--------------------------------")
            for value in event.values():
                # print("Agent:", value["messages"])
                print(value["messages"][-1].content)
            print("--------------------------------")

user_query = "What's the capital of France and what is the hypotenuse length of a triangle with side lengths of 10 and 12?"
stream_tool_responses(user_query)