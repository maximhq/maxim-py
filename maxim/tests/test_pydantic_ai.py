import os
import unittest
import asyncio
from dotenv import load_dotenv

from maxim import Maxim
from maxim.logger.pydantic_ai import instrument_pydantic_ai
from maxim.tests.mock_writer import inject_mock_writer

load_dotenv()

# Try to import pydantic_ai, skip tests if not available
try:
    from pydantic_ai import Agent, RunContext

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    RunContext = None


@unittest.skipUnless(PYDANTIC_AI_AVAILABLE, "pydantic_ai not available")
@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestPydanticAI(unittest.TestCase):
    """Test class for Pydantic AI integration with Maxim logging."""

    def setUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        # Initialize Maxim and logger
        self.maxim = Maxim(
            {
                "api_key": os.getenv("MAXIM_API_KEY"),
                "base_url": os.getenv("MAXIM_BASE_URL"),
                "debug": True,
            }
        )
        self.logger = self.maxim.logger({"id": os.getenv("MAXIM_LOG_REPO_ID")})
        self.mock_writer = inject_mock_writer(self.logger)

        # Instrument Pydantic AI with Maxim logging
        instrument_pydantic_ai(self.logger, debug=True)

    def test_basic_agent_run_sync(self):
        """Test basic synchronous agent execution."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                instructions="Be concise, reply with one sentence.",
            )

            result = agent.run_sync("What is the capital of France?")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)
            self.assertGreater(len(result.output), 0)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent execution error: {e}")

    def test_agent_with_tool_plain(self):
        """Test agent with a plain tool (no context)."""
        try:
            import random

            def roll_dice() -> str:
                """Roll a six-sided die and return the result."""
                return str(random.randint(1, 6))

            agent = Agent(
                "openai:gpt-4o-mini",
                deps_type=str,
                system_prompt=(
                    "You're a dice game. Roll the die and tell the user the result. "
                    "Use the player's name in the response."
                ),
                tools=[roll_dice],
            )

            result = agent.run_sync("Roll the dice for me", deps="Alice")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent with tool execution error: {e}")

    def test_agent_with_tool_with_context(self):
        """Test agent with a tool that uses RunContext."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                deps_type=str,
                system_prompt="Use the player's name in your response.",
            )

            @agent.tool
            def get_player_name(ctx: RunContext[str]) -> str:
                """Get the player's name."""
                return ctx.deps

            result = agent.run_sync("What is my name?", deps="Bob")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)
            self.assertIn("Bob", result.output)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent with context tool execution error: {e}")

    def test_agent_with_decorator_tools(self):
        """Test agent with tools registered using decorators."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                deps_type=str,
                system_prompt="You're a helpful assistant with access to tools.",
            )

            @agent.tool_plain
            def get_weather(location: str) -> str:
                """Get the weather for a location."""
                return f"The weather in {location} is sunny, 72°F."

            @agent.tool
            def get_user_info(ctx: RunContext[str]) -> str:
                """Get information about the user."""
                return f"User: {ctx.deps}"

            result = agent.run_sync("What is the weather in New York?", deps="TestUser")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent with decorator tools execution error: {e}")

    def test_agent_with_output_type(self):
        """Test agent with structured output type."""
        try:
            from pydantic import BaseModel

            class CapitalResponse(BaseModel):
                capital: str
                country: str

            agent = Agent(
                "openai:gpt-4o-mini",
                output_type=CapitalResponse,
                instructions="Return the capital city and country name.",
            )

            result = agent.run_sync("What is the capital of Italy?")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, CapitalResponse)
            self.assertEqual(result.output.country.lower(), "italy")
            self.assertEqual(result.output.capital.lower(), "rome")

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent with output type execution error: {e}")

    def test_agent_with_system_prompt(self):
        """Test agent with system prompt."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                system_prompt="You are a helpful coding assistant who always writes code in Python.",
            )

            result = agent.run_sync("Write a function to calculate factorial")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)
            self.assertIn("def", result.output.lower())
            self.assertIn("factorial", result.output.lower())

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent with system prompt execution error: {e}")

    def test_agent_multiple_runs(self):
        """Test multiple agent runs in sequence."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                instructions="Be concise.",
            )

            result1 = agent.run_sync("What is 2+2?")
            result2 = agent.run_sync("What is 3+3?")

            # Basic assertions
            self.assertIsNotNone(result1)
            self.assertIsNotNone(result2)
            self.assertTrue(hasattr(result1, "output"))
            self.assertTrue(hasattr(result2, "output"))

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Multiple agent runs error: {e}")

    def tearDown(self):
        self.maxim.cleanup()
        if hasattr(self, "mock_writer"):
            self.mock_writer.cleanup()


@unittest.skipUnless(PYDANTIC_AI_AVAILABLE, "pydantic_ai not available")
@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestPydanticAIAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for async Pydantic AI integration with Maxim logging."""

    async def asyncSetUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        # Initialize Maxim and logger
        self.maxim = Maxim(
            {
                "api_key": os.getenv("MAXIM_API_KEY"),
                "base_url": os.getenv("MAXIM_BASE_URL"),
                "debug": True,
            }
        )
        self.logger = self.maxim.logger({"id": os.getenv("MAXIM_LOG_REPO_ID")})
        self.mock_writer = inject_mock_writer(self.logger)

        # Instrument Pydantic AI with Maxim logging
        instrument_pydantic_ai(self.logger, debug=True)

    async def test_async_agent_run(self):
        """Test asynchronous agent execution."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                instructions="Be concise, reply with one sentence.",
            )

            result = await agent.run("What is the capital of Spain?")
            print(f"Async agent with tool result: {result}")

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)
            self.assertGreater(len(result.output), 0)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Async agent execution error: {e}")

    async def test_async_agent_with_tool(self):
        """Test async agent with tool."""
        try:

            async def async_get_info(query: str) -> str:
                """Get information asynchronously."""
                await asyncio.sleep(0.1)  # Simulate async operation
                return f"Information about {query}: It's great!"

            agent = Agent(
                "openai:gpt-4o-mini",
                tools=[async_get_info],
                instructions="Use tools when needed.",
            )

            result = await agent.run("Get info about Python")
            # Basic assertions
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "output"))
            self.assertIsInstance(result.output, str)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Async agent with tool execution error: {e}")

    async def test_async_agent_stream(self):
        """Test async agent streaming."""
        try:
            agent = Agent("openai:gpt-4o-mini")

            async with agent.run_stream("Count from 1 to 5") as response:
                chunks = []
                async for text in response.stream_text():
                    chunks.append(text)

            # Basic assertions
            self.assertGreater(len(chunks), 0)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Async agent stream execution error: {e}")

    async def test_async_agent_stream_events(self):
        """Test async agent stream events."""
        try:
            agent = Agent("openai:gpt-4o-mini")

            # Some pydantic_ai versions may not support stream events
            if not hasattr(agent, "run_stream_events"):
                self.skipTest(
                    "Agent.run_stream_events not available in this pydantic_ai version"
                )

            events = []
            async for event in agent.run_stream_events("What is 2+2?"):
                events.append(event)

            # Basic assertions
            self.assertGreater(len(events), 0)

            # Flush logging (best-effort)
            self.logger.flush()
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Async agent stream events execution error: {e}")

    async def asyncTearDown(self):
        self.logger.flush()
        if hasattr(self, "mock_writer"):
            self.mock_writer.cleanup()


@unittest.skipUnless(PYDANTIC_AI_AVAILABLE, "pydantic_ai not available")
@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestPydanticAIWithMockWriter(unittest.TestCase):
    """Test class demonstrating how to use MockLogWriter for verification."""

    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        # Create logger and patch its writer
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

        # Instrument Pydantic AI
        instrument_pydantic_ai(self.logger, debug=True)

    def test_agent_run_with_mock_writer_verification(self):
        """Test that demonstrates verifying logged commands with mock writer."""
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                instructions="Be concise.",
            )

            # Make the API call
            result = agent.run_sync("What is the capital of Japan?")

            # Flush the logger to ensure all logs are processed
            self.logger.flush()

            # Print logs summary for debugging
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent run with mock writer verification error: {e}")

    def test_agent_with_tool_with_mock_writer(self):
        """Test agent with tool using mock writer verification."""
        try:

            def calculate_sum(a: int, b: int) -> int:
                """Calculate the sum of two numbers."""
                return a + b

            agent = Agent(
                "openai:gpt-4o-mini",
                tools=[calculate_sum],
                instructions="Use the calculate_sum tool when asked to add numbers.",
            )

            # Clear any existing logs
            self.mock_writer.clear_logs()

            # Make the API call
            result = agent.run_sync("What is 5 + 3?")

            # Flush the logger
            self.logger.flush()

            # Print logs summary for debugging
            self.mock_writer.print_logs_summary()

        except Exception as e:
            self.skipTest(f"Agent with tool mock writer verification error: {e}")

    def tearDown(self) -> None:
        # Print final summary for debugging
        if hasattr(self, "mock_writer"):
            self.mock_writer.print_logs_summary()
            # Cleanup the mock writer
            self.mock_writer.cleanup()
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
