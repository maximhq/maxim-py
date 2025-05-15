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

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.INFO)
env = "beta"

openAIKey = data["openAIKey"]
anthropicApiKey = data["anthropicApiKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]
tavilyApiKey = data["tavilyApiKey"]

os.environ["TAVILY_API_KEY"] = tavilyApiKey


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

config = Config(api_key=apiKey, base_url=baseUrl, debug=True)
maxim = Maxim(config)
config = LoggerConfig(id=repoId)
logger = maxim.logger(config)


@langgraph_agent(name="ask_agent")
async def ask_agent(query: str) -> str:
    config = {"recursion_limit": 50, "callbacks": [langchain_callback()]}
    async for event in app.astream(input={"messages": [query]}, config=config):
        for k, v in event.items():
            if k == "agent":
                response = str(v["messages"][0].content)
    return response


@flask_app.post("/chat")
@trace(logger=logger, name="chat_v1", id=lambda: request.headers["reqId"])
async def handle():
    current_trace().add_tag("test", "yes")
    resp = await ask_agent(request.json["query"])
    print(f"answer: {resp}")
    current_trace().set_output(resp)
    return jsonify({"result": resp})


class TestLangGraph(unittest.TestCase):
    def setUp(self) -> None:

        return super().setUp()

    def test_log_agent(self):

        with flask_app.test_client() as client:
            response = client.post(
                "/chat",
                headers={"reqId": str(uuid4())},
                json={"query": "name the last 3 captains of indian cricket team"},
            )
            self.assertEqual(response.status_code, 200)

    def tearDown(self) -> None:
        logger.flush()
        return super().tearDown()
