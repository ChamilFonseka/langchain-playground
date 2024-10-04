from dotenv import load_dotenv
load_dotenv()
#----------------------------------------------------------------------

from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)]
#----------------------------------------------------------------------

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)
#----------------------------------------------------------------------

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages: Annotated[list, add_messages]
#----------------------------------------------------------------------

def chatbot(state: State):
    # return {"messages": [llm.invoke(state["messages"])]}
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
#----------------------------------------------------------------------

from langgraph.graph import StateGraph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
#----------------------------------------------------------------------

from langgraph.prebuilt import ToolNode, tools_condition
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
#----------------------------------------------------------------------  

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
#----------------------------------------------------------------------  

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)
#---------------------------------------------------------------------- 

def stream_graph_updates(user_input: str, chat_id: str):
    config = {"configurable": {"thread_id": chat_id}}
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

    # events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"
    # )
    # for event in events:
    #     event["messages"][-1].pretty_print()
#----------------------------------------------------------------------     

import uuid
myuuid = uuid.uuid4()
#---------------------------------------------------------------------- 

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input, myuuid)
    except Exception as error:
        print("An error occurred: ", error)
        break
#----------------------------------------------------------------------    