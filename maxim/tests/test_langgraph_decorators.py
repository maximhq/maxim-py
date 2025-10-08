"""
Test file demonstrating the @tool_call decorator usage with LangGraph and LangChain.

This file provides comprehensive examples of:
1. Basic @tool_call decorator usage with different return types (dict, str, list)
2. Async tool functions with @tool_call decorator
3. Tool context access using current_tool_call()
4. Integration with LangChain @tool decorator
5. Comparison between Maxim-only and LangChain-only tools
6. Error handling in tool functions
7. Tool call tracking within trace and span contexts

The examples demonstrate how to:
- Track tool calls with metadata (name, description, tags)
- Handle different return types (JSON, strings, structured data)
- Integrate Maxim tracking with existing LangChain tools
- Access tool call context from within decorated functions
- Properly handle errors in tool execution

Run this file to see all examples in action with proper Maxim logging.
"""

import os
from functools import lru_cache
from typing import Annotated, Literal, Sequence, TypedDict
import asyncio

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages

from maxim import Maxim
from maxim.decorators import current_trace, span, trace
from maxim.decorators.langchain import langchain_callback, langgraph_agent
from maxim.decorators.tool_call import tool_call, current_tool_call

from dotenv import load_dotenv
load_dotenv()

# API Keys
openAIKey = os.environ.get("OPENAI_API_KEY", None)
anthropicApiKey = os.environ.get("ANTHROPIC_API_KEY", None)
tavilyApiKey = os.environ.get("TAVILY_API_KEY", None)
workspaceId = os.environ.get("MAXIM_WORKSPACE_ID", None)


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
app = workflow.compile()


# Initialize Maxim (only if not already initialized)
try:
    maxim_client = Maxim()
    logger = maxim_client.logger()
except RuntimeError as e:
    if "already initialized" in str(e):
        # Get the existing instance - accessing protected member is intentional here
        # pylint: disable=protected-access
        logger = Maxim._instance.logger()
    else:
        raise e


# Tool call decorator examples with real functionality
@tool_call(name="file-reader", description="Read and analyze a text file")
def read_file_tool(file_path: str) -> dict:
    """Read a file and return its analysis."""
    try:
        # For demo, we'll create a mock file analysis
        import os
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            analysis = {
                "file_path": file_path,
                "file_size": len(content),
                "line_count": len(content.split('\n')),
                "word_count": len(content.split()),
                "char_count": len(content),
                "preview": content[:100] + "..." if len(content) > 100 else content
            }
        else:
            # Create mock analysis for demo
            analysis = {
                "file_path": file_path,
                "file_size": 1250,
                "line_count": 45,
                "word_count": 200,
                "char_count": 1250,
                "preview": "This is a mock file analysis since the file doesn't exist...",
                "note": "Mock data - file not found"
            }
        
        return analysis
    except Exception as e:
        return {"error": str(e), "file_path": file_path}


@tool_call(name="text-processor", description="Process and transform text")
def process_text_tool(text: str, operation: str = "analyze") -> dict:
    """Process text with various operations."""
    import re
    from collections import Counter
    
    result = {"original_text": text, "operation": operation}
    
    if operation == "analyze":
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        result.update({
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "char_count": len(text),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "most_common_words": dict(Counter(word.lower().strip('.,!?') for word in words).most_common(5))
        })
    
    elif operation == "uppercase":
        result["processed_text"] = text.upper()
    
    elif operation == "reverse":
        result["processed_text"] = text[::-1]
    
    elif operation == "word_count":
        result["word_count"] = len(text.split())
    
    else:
        result["error"] = f"Unknown operation: {operation}"
    
    return result


@tool_call(
    name="data-aggregator", 
    description="Aggregate and summarize numerical data",
    tags={"category": "data-analysis", "version": "v2"}
)
def aggregate_data_tool(numbers: list, operations: list = None) -> dict:
    """Aggregate numerical data with various statistical operations."""
    if not numbers:
        return {"error": "No numbers provided"}
    
    if operations is None:
        operations = ["sum", "mean", "min", "max", "count"]
    
    results = {"input_data": numbers, "operations_performed": operations}
    
    try:
        if "sum" in operations:
            results["sum"] = sum(numbers)
        
        if "mean" in operations:
            results["mean"] = sum(numbers) / len(numbers)
        
        if "min" in operations:
            results["min"] = min(numbers)
        
        if "max" in operations:
            results["max"] = max(numbers)
        
        if "count" in operations:
            results["count"] = len(numbers)
        
        if "median" in operations:
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            results["median"] = (sorted_nums[n//2] + sorted_nums[(n-1)//2]) / 2
        
        if "std_dev" in operations:
            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            results["std_dev"] = variance ** 0.5
        
        return results
    
    except Exception as e:
        return {"error": str(e), "input_data": numbers}


@tool_call(name="async-api-simulator", description="Simulate async API calls with data processing")
async def async_api_simulator(endpoint: str, data: dict = None) -> dict:
    """Simulate async API calls with realistic processing."""
    import json
    import random
    
    # Simulate network delay
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    if endpoint == "user_profile":
        user_id = data.get("user_id", "unknown") if data else "unknown"
        return {
            "endpoint": endpoint,
            "user_id": user_id,
            "profile": {
                "name": f"User_{user_id}",
                "email": f"user_{user_id}@example.com",
                "created_at": "2024-01-15T10:30:00Z",
                "last_login": "2024-01-20T14:22:00Z",
                "preferences": {"theme": "dark", "notifications": True}
            },
            "response_time_ms": random.randint(100, 500)
        }
    
    elif endpoint == "analytics":
        date_range = data.get("date_range", "7d") if data else "7d"
        return {
            "endpoint": endpoint,
            "date_range": date_range,
            "metrics": {
                "page_views": random.randint(1000, 10000),
                "unique_visitors": random.randint(500, 5000),
                "bounce_rate": round(random.uniform(0.2, 0.8), 2),
                "avg_session_duration": random.randint(120, 600)
            },
            "response_time_ms": random.randint(200, 800)
        }
    
    elif endpoint == "weather":
        city = data.get("city", "Unknown") if data else "Unknown"
        return {
            "endpoint": endpoint,
            "city": city,
            "weather": {
                "temperature": random.randint(-10, 35),
                "humidity": random.randint(30, 90),
                "condition": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
                "wind_speed": random.randint(0, 25)
            },
            "response_time_ms": random.randint(50, 200)
        }
    
    else:
        return {
            "endpoint": endpoint,
            "error": "Unknown endpoint",
            "available_endpoints": ["user_profile", "analytics", "weather"]
        }


@tool_call(name="tool-with-context", description="Tool that uses current tool call context")
def tool_with_context(message: str) -> str:
    """Tool that demonstrates accessing current tool call context."""
    current_tool = current_tool_call()
    if current_tool:
        # Using _name as it's the internal attribute for tool call name
        # pylint: disable=protected-access
        return f"Tool '{current_tool._name}' processed: {message}"
    return f"Processed: {message}"


# LangChain tool integration examples
@tool
def langchain_weather_tool(city: str) -> str:
    """Get weather information for a city (mock implementation)."""
    # This is a mock weather tool for demonstration
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C", 
        "tokyo": "Rainy, 20°C",
        "paris": "Partly cloudy, 18°C"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool_call(name="maxim-weather-wrapper", description="Maxim-tracked weather tool")
def maxim_weather_tool(city: str) -> dict:
    """Weather tool wrapped with Maxim tool_call decorator."""
    # Call the original LangChain tool
    weather_result = langchain_weather_tool.invoke({"city": city})
    
    return {
        "city": city,
        "weather": weather_result,
        "source": "langchain_weather_tool",
        "timestamp": "2024-01-01T12:00:00Z"
    }


@tool_call(name="maxim-only-calculator", description="Calculator tool with only Maxim @tool_call decorator")
def maxim_only_calculator(operation: str, a: float, b: float) -> str:
    """Calculator tool with only Maxim @tool_call decorator for simpler usage."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    
    result = operations.get(operation.lower(), "Error: Unknown operation")
    return f"{operation}({a}, {b}) = {result}"


@tool
def langchain_only_calculator(operation: str, a: float, b: float) -> str:
    """Calculator tool with only LangChain @tool decorator."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    
    result = operations.get(operation.lower(), "Error: Unknown operation")
    return f"LangChain: {operation}({a}, {b}) = {result}"


@span(name="another-method-span")
def another_method(query: str) -> str:
    return query


@langgraph_agent(name="movie-agent-v1")
async def ask_agent(query: str) -> str:
    config = {"recursion_limit": 50, "callbacks": [langchain_callback()]}
    response = None
    async for event in app.astream(input={"messages": [query]}, config=config):
        for k, v in event.items():
            if k == "agent":
                response = str(v["messages"][0].content)
    return response


@trace(logger=logger, name="tool-demo-trace", tags={"service": "tool-demo-server", "version": "v1"})
async def demonstrate_tools():
    """Demonstrate various tool call decorator usages."""
    print("=== Tool Call Decorator Demo ===")
    
    # Test basic tool calls
    print("\n1. Testing basic addition tool:")
    add_result = add_numbers(10, 5)
    print(f"Addition result: {add_result}")
    
    print("\n2. Testing multiplication tool:")
    mult_result = multiply_numbers(7, 8)
    print(f"Multiplication result: {mult_result}")
    
    print("\n3. Testing data processing tool:")
    data_result = process_data(["hello", "world", "test"])
    print(f"Data processing result: {data_result}")
    
    print("\n4. Testing async calculator:")
    async_result = await async_calculate("multiply", 12, 4)
    print(f"Async calculation result: {async_result}")
    
    print("\n5. Testing tool with context:")
    context_result = tool_with_context("Hello from tool context!")
    print(f"Context tool result: {context_result}")
    
    print("\n6. Testing LangChain tool integration:")
    weather_result = maxim_weather_tool("New York")
    print(f"Weather tool result: {weather_result}")
    
    print("\n7. Testing Maxim-only calculator:")
    maxim_calc_result = maxim_only_calculator("multiply", 15, 3)
    print(f"Maxim-only calculator result: {maxim_calc_result}")
    
    print("\n8. Testing LangChain-only calculator:")
    # For LangChain tools, we need to use invoke() method with a dict
    langchain_calc_result = langchain_only_calculator.invoke({"operation": "add", "a": 25, "b": 17})
    print(f"LangChain-only calculator result: {langchain_calc_result}")
    
    print("\n9. Testing direct LangChain weather tool:")
    direct_weather = langchain_weather_tool.invoke({"city": "London"})
    print(f"Direct LangChain tool result: {direct_weather}")
    
    # Test error handling
    print("\n10. Testing error handling in async tool:")
    try:
        error_result = await async_calculate("divide", 10, 0)
        print(f"Division by zero result: {error_result}")
    except ZeroDivisionError as e:
        print(f"Division error caught: {e}")
    except ValueError as e:
        print(f"Value error caught: {e}")
    
    # Test error in Maxim-only tool
    print("\n11. Testing error in Maxim-only calculator:")
    error_calc = maxim_only_calculator("divide", 10, 0)
    print(f"Error calculation result: {error_calc}")
    
    # Test error in LangChain-only tool
    print("\n12. Testing error in LangChain-only calculator:")
    langchain_error_calc = langchain_only_calculator.invoke({"operation": "divide", "a": 10, "b": 0})
    print(f"LangChain error calculation result: {langchain_error_calc}")
    
    # Call the original method too
    another_method("Tool demo completed")
    
    # Set trace output and feedback
    current_trace().set_output("Tool call demonstration completed successfully")
    current_trace_obj = current_trace()
    current_trace_obj.feedback({"score": 1, "category": "tool-demo"})
    
    return "Tool demonstration completed"


@trace(logger=logger, name="movie-chat-v1", tags={"service": "movie-chat-v1-server-1"})
async def handle(query: str):
    resp = await ask_agent(query)
    current_trace().set_output(str(resp))
    another_method(str(resp))
    current_trace_obj = current_trace()
    current_trace_obj.feedback({"score": 1})
    return resp


async def main():
    """Main function to run the test"""
    try:
        print("Starting tool call decorator demonstration...")
        
        # First, demonstrate the tool call decorators
        tool_demo_result = await demonstrate_tools()
        print(f"\nTool demo result: {tool_demo_result}")
        
        print("\n" + "="*50)
        print("Now running original movie chat example...")
        
        # Then run the original movie chat example
        resp = await handle("is there any new iron man movies coming this year?")
        print(f"\nMovie chat result: {resp}")
        
    except (RuntimeError, ValueError, ZeroDivisionError) as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
