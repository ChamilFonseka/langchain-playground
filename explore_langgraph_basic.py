from dotenv import load_dotenv
load_dotenv()
#----------------------------------------------------------------------

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
#----------------------------------------------------------------------

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages: Annotated[list, add_messages]
#----------------------------------------------------------------------

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
#----------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END
graph_builder = StateGraph(State)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
#----------------------------------------------------------------------

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        print("An error occurred. Exiting...")
        break
#----------------------------------------------------------------------    