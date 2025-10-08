import os
from functools import lru_cache
from typing import Annotated, Literal, Sequence, TypedDict
import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from maxim import Maxim
from maxim.logger.langchain import MaximLangchainTracer

from dotenv import load_dotenv
load_dotenv()

# API Keys
openAIKey = os.environ.get("OPENAI_API_KEY", None)
anthropicApiKey = os.environ.get("ANTHROPIC_API_KEY", None)
tavilyApiKey = os.environ.get("TAVILY_API_KEY", None)



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [TavilySearchResults(max_results=1, tavily_api_key=tavilyApiKey)]


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=openAIKey)
    elif model_name == "anthropic":
        # Updated to use a current model instead of the deprecated one
        model = ChatAnthropic(
            temperature=0,
            model_name="claude-3-5-sonnet-20241022",  # Updated model name
            api_key=anthropicApiKey,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get("configurable", {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define a new graph - using context_schema instead of deprecated config_schema
workflow = StateGraph(AgentState, context_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")


# Initialize Maxim (only if not already initialized)
try:
    logger = Maxim({}).logger()
except RuntimeError as e:
    if "already initialized" in str(e):
        # Get the existing instance
        logger = Maxim._instance.logger()
    else:
        raise e


# Compile the app and set up tracer
app = workflow.compile()
maxim_langchain_tracer = MaximLangchainTracer(logger)


def another_method(query: str) -> str:
    return query


async def ask_agent(query: str):
    config = {"recursion_limit": 50, "callbacks": [maxim_langchain_tracer]}
    async for event in app.astream(input={"messages": [query]}, config=config):
        for k, v in event.items():
            if k == "agent":
                response = str(v["messages"][0].content)
                yield response


async def handle(query: str):
    async for resp in ask_agent(query):
        another_method(str(resp))
        yield resp


async def main():
    """Main function to run the test"""
    try:
        async for resp in handle("tell me latest football news?"):
            print(resp)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
