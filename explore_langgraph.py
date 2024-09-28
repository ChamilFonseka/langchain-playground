from dotenv import load_dotenv
load_dotenv()

# --Create the agent--
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0, streaming=True)
agent_runnable = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
#--------------------------------------------------------------------------------------

# --Define the graph state--
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
#--------------------------------------------------------------------------------------

# ---Define the nodes---
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor

tool_executor = ToolExecutor(tools)

# Define the agent
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {'agent_outcome': agent_outcome}

# Define the function to execute tools
def execute_tools(data):
    agent_action = data['agent_outcome']
    output = tool_executor.invoke(agent_action)
    return {'intermediate_steps': [(agent_action, str(output))]}

# Define the logic that will be used to determine which conditional edge to do down
def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return 'end'
    return 'continue'
#--------------------------------------------------------------------------------------

# ---Define the graph---
from langgraph.graph import END, StateGraph

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node('agent', run_agent)
workflow.add_node('action', execute_tools)

# Set the entrypoint as 'agent'
# This means that this node is the first one called
workflow.set_entry_point('agent')

# Define the conditional edge
workflow.add_conditional_edges(
    # First, define the starting node
    'agent',
    # Next, pass in the function that will determine which node is called next
    should_continue,
    # Finally, pass in a mapping
    {
        'continue': 'action',
        'end': END,
    }
)

# Now, add a normal edge from 'tools' to 'agent'
# This means that after 'tools' is called, 'agent' will be called next
workflow.add_edge('action', 'agent')

# Finally, compile it
# This compiles it into a LangChain Runnerble
app = workflow.compile()
#--------------------------------------------------------------------------------------

# inputs = {'input': 'What is the capital of France?', 'chat_history': []}
# for s in app.stream(inputs):
#     print(list(s.values())[0])
#     print('---')

print('######################################')
result = app.invoke({'input': 'What is the capital of France?', 'chat_history': []})
print(result)
