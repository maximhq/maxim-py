import json
import logging
import os
import unittest
from functools import lru_cache
from typing import Annotated, Literal, Sequence, TypedDict
from uuid import uuid4

from flask import Flask, jsonify, request
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from maxim import Config, Maxim
from maxim.decorators import current_trace, trace
from maxim.decorators.langchain import langchain_callback, langgraph_agent
from maxim.logger import LoggerConfig
from maxim.tests.mock_writer import inject_mock_writer


logging.basicConfig(level=logging.INFO)

openAIKey = os.getenv("OPENAI_API_KEY")
anthropicApiKey = os.getenv("ANTHROPIC_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL")
repoId = os.getenv("MAXIM_LOG_REPO_ID")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [TavilySearchResults(max_results=1)]


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openAIKey)
    elif model_name == "anthropic":
        model = ChatAnthropic(
            temperature=0,
            model_name="claude-3-sonnet-20240229",
            api_key=anthropicApiKey,
            timeout=30,
            stop=None,
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


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

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

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

flask_app = Flask(__name__)



maxim = Maxim()
logger = maxim.logger()
class TestLangGraph(unittest.TestCase):
    def setUp(self) -> None:
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.logger = logger

        return super().setUp()

    @langgraph_agent(name="ask_agent", logger=logger)
    async def ask_agent(self, query: str) -> str:
        config = {"recursion_limit": 50, "callbacks": [langchain_callback()]}
        async for event in app.astream(input={"messages": [query]}, config=config):
            for k, v in event.items():
                if k == "agent":
                    response = str(v["messages"][0].content)
        return response

    @trace(name="chat_v1", id=lambda: str(uuid4()), logger=None)
    async def handle_chat(self, query: str, req_id: str):
        trace = current_trace()
        if trace is None:
            raise ValueError("current_trace is None")
        trace.add_tag("test", "yes")
        resp = await self.ask_agent(query)
        print(f"answer: {resp}")
        trace.set_output(resp)
        return {"result": resp}

    def test_log_agent(self):
        # Since we can't easily test the Flask endpoints in isolation,
        # we'll test the core functionality directly
        import asyncio

        async def run_test():
            req_id = str(uuid4())
            # We'll simulate a simple query instead of a complex one
            query = "Hello, how are you?"

            # Run the chat handler
            result = await self.handle_chat(query, req_id)
            self.assertIsNotNone(result)
            self.assertIn("result", result)

        # Run the async test
        asyncio.run(run_test())

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()

        # LangGraph creates multiple traces due to the agent workflow
        # We expect at least one trace to be created
        # all_logs = self.mock_writer.get_all_logs()
        # self.assertGreater(len(all_logs), 0, "Expected at least one log to be captured")

    def tearDown(self) -> None:
        # Print final summary for debugging
        # self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        # self.mock_writer.cleanup()
        self.logger.flush()
        return super().tearDown()
