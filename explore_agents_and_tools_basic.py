from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%I:%M:%p")

# List of tools to be used by the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
prompt = hub.pull("hwchase17/react")

# Initialize the chat model
llm = ChatOpenAI(
    model="gpt-4o", temperature=0
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test input
response = agent_executor.invoke({"input": "What time is it?"})

print(response)




