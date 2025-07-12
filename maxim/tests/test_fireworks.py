import os
import unittest
from uuid import uuid4

import dotenv
from fireworks import LLM

from maxim import Maxim
from maxim.logger.fireworks import instrument_fireworks

dotenv.load_dotenv()

fireworksApiKey = os.getenv("FIREWORKS_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"
repoId = os.getenv("MAXIM_LOG_REPO_ID")

class TestFireworks(unittest.TestCase):
    """Test class for Fireworks AI integration with Maxim logging."""

    def setUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not fireworksApiKey:
            raise ValueError("FIREWORKS_API_KEY environment variable is not set")

        self.logger = Maxim().logger()
        
        # Initialize Fireworks LLM client with instrumentation
        self.llm = LLM(
            model="llama4-maverick-instruct-basic", 
            deployment_type="serverless", 
            api_key=fireworksApiKey
        )
        instrument_fireworks(self.logger)

    def test_non_streaming_chat_completion(self):
        """Test non-streaming chat completion with basic user message"""
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
            max_tokens=150,
            temperature=0.7
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Non-streaming response:", response.choices[0].message.content[:100] + "...")
        
    def test_non_streaming_with_system_message(self):
        """Test non-streaming chat completion with system message"""
        response = self.llm.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant. Always provide concise, practical advice."},
                {"role": "user", "content": "What's the best way to get around San Francisco?"}
            ],
            max_tokens=150,
            temperature=0.5,
            extra_headers={
                "x-maxim-trace-id": str(uuid4()),
                "x-maxim-generation-name": "travel_assistant",
                "x-maxim-generation-tags": {
                    "test_type": "system_message",
                    "city": "san_francisco"
                }
            }
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("System message response:", response.choices[0].message.content[:100] + "...")

    def test_streaming_chat_completion(self):
        """Test streaming chat completion"""
        stream = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": "Tell me a short story about a robot learning to paint."}],
            max_tokens=200,
            temperature=0.8,
            stream=True
        )
        
        # Consume the stream
        full_response = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    full_response += content
                    print(content, end="", flush=True)
        
        print()  # New line after streaming
        # Verify we received streaming data
        self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
        self.assertGreater(len(full_response), 0, "Expected non-empty response")
        print(f"Streaming completed with {chunk_count} chunks")

    def test_streaming_with_system_message(self):
        """Test streaming chat completion with system message"""
        stream = self.llm.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a creative writing assistant. Write in a poetic, descriptive style."},
                {"role": "user", "content": "Describe a sunset over the ocean in 3 sentences."}
            ],
            max_tokens=100,
            temperature=0.9,
            stream=True,
            extra_headers={
                "x-maxim-generation-name": "creative_writing",
                "x-maxim-generation-tags": {
                    "test_type": "streaming_system_message",
                    "style": "poetic"
                }
            }
        )
        
        # Consume the stream
        full_response = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    full_response += content
                    print(content, end="", flush=True)
        
        # Verify we received streaming data
        self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
        self.assertGreater(len(full_response), 0, "Expected non-empty response")
        print(f"System message streaming completed with {chunk_count} chunks")

    def test_different_model_code_completion(self):
        """Test with a different model for code generation"""
        # Create a new LLM instance with a different model
        code_llm = LLM(
            model="llama3-8b-instruct", 
            deployment_type="serverless", 
            api_key=fireworksApiKey
        )
        
        response = code_llm.chat.completions.create(
            messages=[{"role": "user", "content": "Write a Python function to calculate the factorial of a number."}],
            max_tokens=200,
            temperature=0.3
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Code completion response:", response.choices[0].message.content[:100] + "...")

    def test_conversation_context(self):
        """Test multi-turn conversation"""
        messages = [
            {"role": "user", "content": "I'm planning a trip to Japan. What should I pack?"},
        ]
        
        # First response
        response1 = self.llm.chat.completions.create(
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        
        # Verify first response
        self.assertIsNotNone(response1)
        self.assertIsNotNone(response1.choices[0].message.content)
        print("First response:", response1.choices[0].message.content[:100] + "...")
        
        # Add assistant response and continue conversation
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})
        messages.append({"role": "user", "content": "What about specific items for visiting temples?"})
        
        response2 = self.llm.chat.completions.create(
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        
        # Verify second response
        self.assertIsNotNone(response2)
        self.assertIsNotNone(response2.choices[0].message.content)
        print("Second response:", response2.choices[0].message.content[:100] + "...")

    def test_custom_parameters(self):
        """Test with custom Fireworks-specific parameters"""
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            extra_headers={
                "x-maxim-generation-name": "quantum_explanation",
                "x-maxim-generation-tags": {
                    "test_type": "custom_parameters",
                    "topic": "quantum_computing"
                }
            }
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Custom parameters response:", response.choices[0].message.content[:100] + "...")

    def test_trace_and_generation_headers(self):
        """Test custom trace and generation headers"""
        custom_trace_id = str(uuid4())
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=50,
            temperature=0.1,
            extra_headers={
                "x-maxim-trace-id": custom_trace_id,
                "x-maxim-generation-name": "geography_question",
                "x-maxim-generation-tags": {
                    "test_type": "custom_headers",
                    "subject": "geography"
                },
                "x-maxim-trace-tags": {
                    "session_type": "educational",
                    "user_level": "beginner"
                }
            }
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Custom headers response:", response.choices[0].message.content[:100] + "...")

    def tearDown(self) -> None:
        # Clean up any resources if needed
        pass


class TestFireworksAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for Fireworks AI asynchronous integration with Maxim logging."""

    async def asyncSetUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not fireworksApiKey:
            raise ValueError("FIREWORKS_API_KEY environment variable is not set")

        self.logger = Maxim().logger()
        
        # Initialize Fireworks LLM client with instrumentation
        self.llm = LLM(
            model="llama4-maverick-instruct-basic", 
            deployment_type="serverless", 
            api_key=fireworksApiKey
        )
        instrument_fireworks(self.logger)

    async def test_async_non_streaming_chat_completion(self):
        """Test asynchronous non-streaming chat completion"""
        response = await self.llm.chat.completions.acreate(
            messages=[{"role": "user", "content": "What are the benefits of renewable energy?"}],
            max_tokens=150,
            temperature=0.7
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Async non-streaming response:", response.choices[0].message.content[:100] + "...")

    async def test_async_non_streaming_with_system_message(self):
        """Test asynchronous non-streaming chat completion with system message"""
        response = await self.llm.chat.completions.acreate(
            messages=[
                {"role": "system", "content": "You are an environmental expert. Provide scientific and factual information."},
                {"role": "user", "content": "How does solar energy work?"}
            ],
            max_tokens=150,
            temperature=0.5,
            extra_headers={
                "x-maxim-trace-id": str(uuid4()),
                "x-maxim-generation-name": "environmental_expert",
                "x-maxim-generation-tags": {
                    "test_type": "async_system_message",
                    "topic": "solar_energy"
                }
            }
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Async system message response:", response.choices[0].message.content[:100] + "...")

    async def test_async_streaming_chat_completion(self):
        """Test asynchronous streaming chat completion"""
        stream = await self.llm.chat.completions.acreate(
            messages=[{"role": "user", "content": "Write a haiku about artificial intelligence."}],
            max_tokens=100,
            temperature=0.8,
            stream=True
        )
        
        # Consume the stream
        full_response = ""
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    full_response += content
                    print(content, end="", flush=True)
        
        print()  # New line after streaming
        # Verify we received streaming data
        self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
        self.assertGreater(len(full_response), 0, "Expected non-empty response")
        print(f"Async streaming completed with {chunk_count} chunks")

    async def test_async_streaming_with_system_message(self):
        """Test asynchronous streaming chat completion with system message"""
        stream = await self.llm.chat.completions.acreate(
            messages=[
                {"role": "system", "content": "You are a poetry expert. Write elegant and meaningful verse."},
                {"role": "user", "content": "Create a limerick about space exploration."}
            ],
            max_tokens=100,
            temperature=0.9,
            stream=True,
            extra_headers={
                "x-maxim-generation-name": "poetry_expert",
                "x-maxim-generation-tags": {
                    "test_type": "async_streaming_system_message",
                    "genre": "limerick"
                }
            }
        )
        
        # Consume the stream
        full_response = ""
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    full_response += content
                    print(content, end="", flush=True)
        
        print()  # New line after streaming
        # Verify we received streaming data
        self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
        self.assertGreater(len(full_response), 0, "Expected non-empty response")
        print(f"Async system message streaming completed with {chunk_count} chunks")

    async def test_async_conversation_context(self):
        """Test asynchronous multi-turn conversation"""
        messages = [
            {"role": "user", "content": "What are the main programming paradigms?"},
        ]
        
        # First async response
        response1 = await self.llm.chat.completions.acreate(
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        
        # Verify first response
        self.assertIsNotNone(response1)
        self.assertIsNotNone(response1.choices[0].message.content)
        print("First async response:", response1.choices[0].message.content[:100] + "...")
        
        # Add assistant response and continue conversation
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})
        messages.append({"role": "user", "content": "Can you explain object-oriented programming in detail?"})
        
        response2 = await self.llm.chat.completions.acreate(
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        
        # Verify second response
        self.assertIsNotNone(response2)
        self.assertIsNotNone(response2.choices[0].message.content)
        print("Second async response:", response2.choices[0].message.content[:100] + "...")

    async def test_async_custom_parameters(self):
        """Test asynchronous completion with custom parameters"""
        response = await self.llm.chat.completions.acreate(
            messages=[{"role": "user", "content": "Explain machine learning algorithms briefly."}],
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            extra_headers={
                "x-maxim-generation-name": "ml_explanation",
                "x-maxim-generation-tags": {
                    "test_type": "async_custom_parameters",
                    "topic": "machine_learning"
                }
            }
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertTrue(len(response.choices) > 0)
        self.assertIsNotNone(response.choices[0].message.content)
        print("Async custom parameters response:", response.choices[0].message.content[:100] + "...")

    async def asyncTearDown(self) -> None:
        # Clean up any resources if needed
        pass


def run_sync_tests():
    """Run synchronous tests"""
    print("=" * 60)
    print("Running Fireworks Synchronous Tests")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFireworks)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_async_tests():
    """Run asynchronous tests"""
    print("=" * 60)
    print("Running Fireworks Asynchronous Tests")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFireworksAsync)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_all_tests():
    """Run all tests"""
    print("Starting Fireworks Integration Tests")
    print("=" * 60)
    
    sync_success = run_sync_tests()
    async_success = run_async_tests()
    
    print("=" * 60)
    if sync_success and async_success:
        print("✅ All Fireworks tests passed!")
    else:
        print("❌ Some Fireworks tests failed!")
        if not sync_success:
            print("  - Synchronous tests failed")
        if not async_success:
            print("  - Asynchronous tests failed")
    print("=" * 60)
    
    return sync_success and async_success


if __name__ == "__main__":
    # Run all tests when script is executed directly
    success = run_all_tests()
    exit(0 if success else 1)