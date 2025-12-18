import os
import uuid
from dotenv import load_dotenv
import functools
from maxim import Maxim
from maxim.logger.pydantic_ai import instrument_pydantic_ai
from pydantic_ai import Agent, RunContext
import asyncio


# Load environment variables
load_dotenv()

# Set up Maxim logger
maxim = Maxim({"base_url": os.getenv("MAXIM_BASE_URL")})
maxim_logger = maxim.logger()
session_id = str(uuid.uuid4())


def get_maxim_model_settings() -> dict:
    """Return model_settings with Maxim metadata in extra_body for testing."""
    return {
        "temperature": 0.2,
        "max_tokens": 512,
        "extra_body": {
            "maxim_metadata": {
                "session_id": session_id,
                "session_tags": {
                    "env": "staging",
                    "feature": "pydantic_agent_examples",
                },
                "trace_tags": {
                    "suite": "test_pydantic_agent",
                },
            },
        },
    }


# Instrument Pydantic AI once at the top
print("Initializing Maxim instrumentation for Pydantic AI...")
instrument_pydantic_ai(maxim_logger, debug=True)
print("Instrumentation complete!")

# Import Pydantic AI components


def create_simple_agent():
    """Create a simple Pydantic AI agent with math tools."""
    agent = Agent(
        model="openai:gpt-4o-mini",
        name="Simple Math Agent",
        instructions="You are a helpful assistant that can perform calculations.",
    )

    @agent.tool
    def add_numbers(ctx: RunContext, a: float, b: float) -> float:
        """Add two numbers together."""
        print(f"[Tool] Adding {a} + {b}")
        return a + b

    @agent.tool
    def multiply_numbers(ctx: RunContext, a: float, b: float) -> float:
        """Multiply two numbers together."""
        print(f"[Tool] Multiplying {a} * {b}")
        return a * b

    return agent


async def run_simple_example():
    """Run the simple agent example."""
    print("=== Simple Math Agent Example ===")

    # Create the agent
    agent = create_simple_agent()

    # Run multiple calculations
    print("Running first calculation...")
    result = await agent.run(
        "What is 15 + 27?", model_settings=get_maxim_model_settings()
    )
    print(f"Result: {result}")

    print("Running second calculation...")
    result = await agent.run(
        "Calculate 8 * 12", model_settings=get_maxim_model_settings()
    )
    print(f"Result: {result}")

    print("Running third calculation...")
    result = await agent.run(
        "What is 25 + 17 and then multiply that result by 3?",
        model_settings=get_maxim_model_settings(),
    )
    print(f"Result: {result}")

    print("Simple agent example completed!")


from typing import List


def create_advanced_agent():
    """Create an advanced Pydantic AI agent with complex tools."""
    agent = Agent(
        model="openai:gpt-4o-mini",
        name="Advanced Analysis Agent",
        instructions="You are an advanced assistant that can perform various analysis tasks.",
    )

    @agent.tool
    def analyze_text(ctx: RunContext, text: str) -> dict:
        """Analyze text and return statistics."""
        print(f"[Tool] Analyzing text: {text[:50]}...")
        words = text.split()
        return {
            "word_count": len(words),
            "character_count": len(text),
            "average_word_length": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
        }

    @agent.tool
    def generate_list(ctx: RunContext, topic: str, count: int = 5) -> List[str]:
        """Generate a list of items related to a topic."""
        print(f"[Tool] Generating list of {count} items for topic: {topic}")
        return [f"{topic} item {i+1}" for i in range(count)]

    @agent.tool
    def calculate_statistics(ctx: RunContext, numbers: List[float]) -> dict:
        """Calculate basic statistics for a list of numbers."""
        print(f"[Tool] Calculating statistics for {len(numbers)} numbers")
        if not numbers:
            return {"error": "No numbers provided"}

        return {
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "count": len(numbers),
        }

    return agent


async def run_advanced_example():
    """Run the advanced agent example."""
    print("=== Advanced Analysis Agent Example ===")

    # Create the agent
    agent = create_advanced_agent()

    # Run text analysis
    print("Running text analysis...")
    result = await agent.run(
        "Analyze this text: 'The quick brown fox jumps over the lazy dog.'",
        model_settings=get_maxim_model_settings(),
    )
    print(f"Text Analysis Result: {result}")

    # Run list generation
    print("Running list generation...")
    result = await agent.run(
        "Generate a list of 3 programming languages",
        model_settings=get_maxim_model_settings(),
    )
    print(f"List Generation Result: {result}")

    # Run statistics calculation
    print("Running statistics calculation...")
    result = await agent.run(
        "Calculate statistics for the numbers [10, 20, 30, 40, 50]",
        model_settings=get_maxim_model_settings(),
    )
    print(f"Statistics Result: {result}")

    # Run combined task
    print("Running combined task...")
    result = await agent.run(
        "Analyze the text 'Python is awesome' and then generate a list of 2 related programming concepts",
        model_settings=get_maxim_model_settings(),
    )
    print(f"Combined Task Result: {result}")

    print("Advanced agent example completed!")


def create_streaming_agent():
    """Create a Pydantic AI agent with streaming capabilities."""
    agent = Agent(
        model="openai:gpt-4o-mini",
        name="Streaming Agent",
        instructions="You are a helpful assistant that can provide detailed explanations and perform calculations.",
    )

    @agent.tool
    def add_numbers(ctx: RunContext, a: float, b: float) -> float:
        """Add two numbers together."""
        print(f"[Tool] Adding {a} + {b}")
        return a + b

    @agent.tool
    def multiply_numbers(ctx: RunContext, a: float, b: float) -> float:
        """Multiply two numbers together."""
        print(f"[Tool] Multiplying {a} * {b}")
        return a * b

    @agent.tool
    def explain_concept(ctx: RunContext, topic: str) -> str:
        """Provide a brief explanation of a concept."""
        print(f"[Tool] Explaining concept: {topic}")
        return f"Here's a brief explanation of {topic}: It's an important concept in its field."

    return agent


async def run_streaming_example():
    """Run the streaming agent example."""
    print("=== Streaming Agent Example ===")

    # Create the agent
    agent = create_streaming_agent()

    # Use streaming mode for detailed explanations
    print("Running streaming explanation...")
    async with agent.run_stream(
        "Explain what is 2 + 2 in detail and then calculate 5 * 6",
        model_settings=get_maxim_model_settings(),
    ) as stream:
        print("Streaming in progress...")
        # The stream will complete automatically when the context manager exits
        print("Streaming completed")

    # Run another streaming example
    print("Running streaming concept explanation...")
    async with agent.run_stream(
        "Explain the concept of machine learning and then add 10 + 15",
        model_settings=get_maxim_model_settings(),
    ) as stream:
        print("Streaming concept explanation in progress...")
        print("Streaming concept explanation completed")

    # Run a simple calculation without blocking the event loop
    print("Running calculation for comparison...")
    result = await agent.run(
        "What is 7 * 8?", model_settings=get_maxim_model_settings()
    )
    print(f"Result: {result}")

    print("Streaming agent example completed!")


async def run_additional_example():
    """Run an additional agent example."""
    print("=== Additional Agent Example ===")

    agent = create_streaming_agent()

    # Run multiple operations
    print("Running first calculation...")
    result = await agent.run(
        "What is 12 + 18?", model_settings=get_maxim_model_settings()
    )
    print(f"Result: {result}")

    print("Running second calculation...")
    result = await agent.run(
        "Calculate 6 * 9", model_settings=get_maxim_model_settings()
    )
    print(f"Result: {result}")

    print("Example completed!")


async def main():
    """Main function to run all examples."""
    print("Pydantic AI Integration with Maxim - Complete Example")
    print("=" * 55)

    # Run simple agent example
    await run_simple_example()

    print("\n" + "=" * 55)

    # Run advanced agent example
    await run_advanced_example()

    print("\n" + "=" * 55)

    # Run streaming example
    await run_streaming_example()

    print("\n" + "=" * 55)

    # Run additional example
    await run_additional_example()

    print("\n=== All Examples Completed ===")
    print("Check your Maxim dashboard to see:")
    print("- Agent traces with tool calls")
    print("- Model generations")
    print("- Performance metrics")
    print("- Streaming responses")


# Run all examples
asyncio.run(main())

# Explicitly end the Maxim session used by the Pydantic AI instrumentation
try:
    if hasattr(instrument_pydantic_ai, "end_session"):
        # Best-effort: end any existing session without creating a new one
        instrument_pydantic_ai.end_session()
    maxim_logger.flush()
except Exception:
    # Best-effort cleanup for tests
    pass
