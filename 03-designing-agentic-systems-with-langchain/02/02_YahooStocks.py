import os
import math 

from typing import Annotated 
from typing_extensions import TypedDict 



from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import WikipediaQueryRun

from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llmOpenAI = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

api_wrapper = WikipediaAPIWrapper(top_k_results=1) 
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper) 
yahoo_finance_tool = YahooFinanceNewsTool( top_k=2)
my_tools = [wikipedia_tool, yahoo_finance_tool] 

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

try:
    img = graph.get_graph().draw_mermaid_png()
    with open("graph_visualization_YahooStocks.png", "wb") as f:
        f.write(img)
    print("Visualization saved as 'graph_visualization_YahooStocks.png'")
except Exception as e:
    print(f"Error generating visualization: {e}")

# Create a variable to store conversation state
conversation_state = {"messages": []}

def stream_tool_responses(user_input: str, maintain_state=True):
    global conversation_state
    
    # If we're maintaining state, use existing conversation
    # Otherwise start a new conversation
    if maintain_state and conversation_state["messages"]:
        # Create a copy of the current state to work with
        current_state = {"messages": conversation_state["messages"].copy()}
        # Add the new user message
        current_state["messages"].append(("user", user_input))
    else:
        # Start a fresh conversation
        current_state = {"messages": [("user", user_input)]}
        # Reset the global state too
        conversation_state = {"messages": []}
    
    # Create an output file for this query
    output_file = f"agent_response_{user_input.replace(' ', '_').replace('?', '').lower()}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Process through the graph
        result = graph.invoke(current_state)
        last_message = result["messages"][-1]
        # Write the response to file
        f.write(f"{last_message.content}\n")
        
        # Update our conversation state with the result
        conversation_state = result
    
    print(f"Response saved to: {output_file}")


# First question - starts a new conversation
stream_tool_responses("Tell me about Microsoft and Apple", maintain_state=False)

# Second question - follows up on the first conversation
stream_tool_responses("Analyze their stock prices", maintain_state=True)


def interactive_chat():
    state = {"messages": []}
    print("Chat with the agent (type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Add user message to state
        state["messages"].append(("user", user_input))
        
        # Process through graph
        result = graph.invoke(state)
        
        # Update state with result
        state = result
        
        # Print agent's response
        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content}")

# Run the interactive chat
# interactive_chat()