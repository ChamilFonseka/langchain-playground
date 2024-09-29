from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Chatbot:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI()
        self._build_graph()
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        graph_builder = StateGraph(State)
        graph_builder.add_node('chatbot', self.chatbot)
        graph_builder.add_edge(START, 'chatbot')
        graph_builder.add_edge('chatbot', END)
        return graph_builder.compile()

    def chatbot(self, state: State) -> State:
        return {'messages': [self.llm.invoke(state['messages'])]}

    def stream_graph_updates(self, user_input: str):
        for event in self.graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    def run(self):
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.stream_graph_updates(user_input)
            except:
                user_input = "What do you know about LangGraph?"
                print("User: " + user_input)
                self.stream_graph_updates(user_input)
                break

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.run()