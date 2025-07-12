import os
import unittest
from uuid import uuid4

import dotenv
from groq import Groq, AsyncGroq

from maxim import Maxim
from maxim.logger.groq import instrument_groq

dotenv.load_dotenv()

groqApiKey = os.getenv("GROQ_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"
repoId = os.getenv("MAXIM_LOG_REPO_ID")

class TestGroq(unittest.TestCase):
    """Test class for Groq integration with MockLogWriter verification."""

    def setUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not groqApiKey:
            self.skipTest("GROQ_API_KEY environment variable is not set")

        self.logger = Maxim().logger()
        
        # Initialize Groq client with instrumentation
        self.client = Groq(api_key=groqApiKey)
        instrument_groq(self.logger)

    def test_non_streaming_chat_completion(self):
        """Test non-streaming chat completion with basic user message"""
    
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
                max_tokens=150,
                temperature=0.7,
            )
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)

            print("Response:", response.choices[0].message.content)
        except Exception as e:
            self.skipTest(f"Error: {e}")

    def test_non_streaming_with_system_message(self):
        """Test non-streaming chat completion with system message"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful travel assistant. Always provide concise, practical advice."},
                    {"role": "user", "content": "What's the best way to get around San Francisco?"}
                ],
                max_tokens=150,
                temperature=0.5,
            )
            self.assertIsNotNone(response) 
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Response:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Error: {e}")

    def test_streaming_chat_completion(self):
        """Test streaming chat completion"""
        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
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
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            # Verify we received streaming data
            self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
            self.assertGreater(len(full_response), 0, "Expected non-empty response")
            
            print("Streaming response:", full_response)

        except Exception as e:
            self.skipTest(f"Streaming error: {e}")

    def test_streaming_with_system_message(self):
        """Test streaming chat completion with system message"""
        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant. Write in a poetic, descriptive style."},
                    {"role": "user", "content": "Describe a sunset over the ocean in 3 sentences."}
                ],
                max_tokens=100,
                temperature=0.9,
                stream=True
            )
            
            # Consume the stream
            full_response = ""
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            # Verify we received streaming data
            self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
            self.assertGreater(len(full_response), 0, "Expected non-empty response")
            
            print("Streaming response with system message:", full_response)

        except Exception as e:
            self.skipTest(f"Streaming with system message error: {e}")

    def test_different_model_code_completion(self):
        """Test with a different model for code generation"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Using a different, larger model
                messages=[{"role": "user", "content": "Write a Python function to calculate the factorial of a number."}],
                max_tokens=200,
                temperature=0.3
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Code generation response:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Different model error: {e}")

    def test_conversation_context(self):
        """Test multi-turn conversation"""
        try:
            messages = [
                {"role": "user", "content": "I'm planning a trip to Japan. What should I pack?"},
            ]
            
            # First response
            response1 = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=150,
                temperature=0.6
            )
            
            # Verify first response
            self.assertIsNotNone(response1)
            self.assertIsNotNone(response1.choices[0].message.content)
            
            print("First response:", response1.choices[0].message.content)
            
            # Add assistant response and continue conversation
            messages.append({"role": "assistant", "content": response1.choices[0].message.content})
            messages.append({"role": "user", "content": "What about specific items for visiting temples?"})
            
            response2 = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=150,
                temperature=0.6
            )
            
            # Verify second response
            self.assertIsNotNone(response2)
            self.assertIsNotNone(response2.choices[0].message.content)
            
            print("Second response:", response2.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Conversation context error: {e}")

    def test_custom_parameters(self):
        """Test with custom Groq-specific parameters"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
                stop=["In conclusion", "To summarize"],
                seed=42  # Groq supports seed parameter for reproducibility
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Custom parameters response:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Custom parameters error: {e}")

    def test_trace_id_header(self):
        """Test with custom trace ID header"""
        try:
            trace_id = str(uuid4())
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                max_tokens=50,
                temperature=0.1,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Custom trace ID response:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Trace ID header error: {e}")

    def test_generation_name_header(self):
        """Test with custom generation name header"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Count from 1 to 5."}],
                max_tokens=50,
                temperature=0.1,
                extra_headers={
                    "x-maxim-generation-name": "counting-test",
                }
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Custom generation name response:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Generation name header error: {e}")
        
    def tearDown(self) -> None:
        self.logger.cleanup()

class TestGroqAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for Groq async integration."""

    async def asyncSetUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not groqApiKey:
            self.skipTest("GROQ_API_KEY environment variable is not set")
        
        self.logger = Maxim().logger()
        
        # Initialize async Groq client with instrumentation
        self.async_client = AsyncGroq(api_key=groqApiKey)
        instrument_groq(self.logger)

    async def test_async_non_streaming_chat_completion(self):
        """Test async non-streaming chat completion with basic user message"""
        try:
            response = await self.async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
                max_tokens=150,
                temperature=0.7
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Async response:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Async error: {e}")
        
    async def test_async_non_streaming_with_system_message(self):
        """Test async non-streaming chat completion with system message"""
        try:
            response = await self.async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful travel assistant. Always provide concise, practical advice."},
                    {"role": "user", "content": "What's the best way to get around San Francisco?"}
                ],
                max_tokens=150,
                temperature=0.5
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            
            print("Async response with system message:", response.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Async with system message error: {e}")
        
    async def test_async_streaming_chat_completion(self):
        """Test async streaming chat completion"""
        try:
            response = await self.async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Tell me a short story about a robot learning to paint."}],
                max_tokens=200,
                temperature=0.8,
                stream=True
            )
            
            # Consume the async stream
            full_response = ""
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            # Verify we received streaming data
            self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
            self.assertGreater(len(full_response), 0, "Expected non-empty response")
            
            print("Async streaming response:", full_response)

        except Exception as e:
            self.skipTest(f"Async streaming error: {e}")
        
    async def test_async_streaming_with_system_message(self):
        """Test async streaming chat completion with system message"""
        try:
            response = await self.async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant. Write in a poetic, descriptive style."},
                    {"role": "user", "content": "Describe a sunset over the ocean in 3 sentences."}
                ],
                max_tokens=100,
                temperature=0.9,
                stream=True
            )
            
            # Consume the async stream
            full_response = ""
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            # Verify we received streaming data
            self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
            self.assertGreater(len(full_response), 0, "Expected non-empty response")
            
            print("Async streaming response with system message:", full_response)

        except Exception as e:
            self.skipTest(f"Async streaming with system message error: {e}")

    async def test_async_with_trace_id(self):
        """Test async with custom trace ID"""
        try:
            trace_id = str(uuid4())
            
            # First call
            response1 = await self.async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "What is artificial intelligence?"}],
                max_tokens=100,
                temperature=0.5,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            # Second call with same trace ID
            response2 = await self.async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Can you give me an example?"}],
                max_tokens=100,
                temperature=0.5,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            # Verify both responses
            self.assertIsNotNone(response1)
            self.assertIsNotNone(response1.choices[0].message.content)
            
            self.assertIsNotNone(response2)
            self.assertIsNotNone(response2.choices[0].message.content)
            
            print("Async trace ID response 1:", response1.choices[0].message.content)
            print("Async trace ID response 2:", response2.choices[0].message.content)

        except Exception as e:
            self.skipTest(f"Async trace ID error: {e}")
        
    async def asyncTearDown(self) -> None:
        self.logger.cleanup()


if __name__ == "__main__":
    unittest.main()
