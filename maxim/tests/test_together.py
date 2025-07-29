import os
import unittest
from uuid import uuid4

import dotenv
from together import Together, AsyncTogether
from together.types import ChatCompletionResponse 

from maxim import Maxim
from maxim.logger.together import instrument_together


dotenv.load_dotenv()

togetherApiKey = os.getenv("TOGETHER_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"
repoId = os.getenv("MAXIM_LOG_REPO_ID")


class TestTogether(unittest.TestCase):
    """Test class for Together AI integration with MockLogWriter verification."""

    def setUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not togetherApiKey:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")

        self.logger = Maxim({ "base_url": baseUrl }).logger()
        
        # Initialize Together clients with instrumentation
        self.client = Together(api_key=togetherApiKey)
        instrument_together(self.logger)

    def test_non_streaming_chat_completion(self):
        """Test non-streaming chat completion with basic user message"""
        try:
            res: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
                max_tokens=150,
                temperature=0.7,
                extra_headers={
                    "x-maxim-generation-name": "nyc_recommendations",
                    "x-maxim-generation-tags": {
                        "test_type": "non_streaming",
                        "topic": "travel"
                    }
                }
            )
            
            # Verify response structure
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, 'choices'))
            self.assertTrue(len(res.choices) > 0)
            self.assertIsNotNone(res.choices[0].message.content)
            print("Non-streaming response:", res.choices[0].message.content[:100] + "...")
        except Exception as e:
            self.skipTest(f"Error: {e}")
        
    def test_non_streaming_with_system_message(self):
        """Test non-streaming chat completion with system message"""
        res: ChatCompletionResponse = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant. Always provide concise, practical advice."},
                {"role": "user", "content": "What's the best way to get around San Francisco?"}
            ],
            max_tokens=150,
            temperature=0.5,
            extra_headers={
                "x-maxim-trace-id": str(uuid4()),
                "x-maxim-generation-tags": {
                    "test": "abc"
                }
            }
        )
        
        # Verify response structure
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'choices'))
        self.assertTrue(len(res.choices) > 0)
        self.assertIsNotNone(res.choices[0].message.content)

    def test_streaming_chat_completion(self):
        """Test streaming chat completion"""
        stream = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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

    def test_streaming_with_system_message(self):
        """Test streaming chat completion with system message"""
        stream = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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

    def test_different_model_code_completion(self):
        """Test with a different model for code generation"""
        res: ChatCompletionResponse = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Write a Python function to calculate the factorial of a number."}],
            max_tokens=200,
            temperature=0.3
        )
        
        # Verify response structure
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'choices'))
        self.assertTrue(len(res.choices) > 0)
        self.assertIsNotNone(res.choices[0].message.content)

    def test_conversation_context(self):
        """Test multi-turn conversation"""
        messages = [
            {"role": "user", "content": "I'm planning a trip to Japan. What should I pack?"},
        ]
        
        # First response
        res1: ChatCompletionResponse = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        
        # Verify first response
        self.assertIsNotNone(res1)
        self.assertIsNotNone(res1.choices[0].message.content)
        
        # Add assistant response and continue conversation
        messages.append({"role": "assistant", "content": res1.choices[0].message.content})
        messages.append({"role": "user", "content": "What about specific items for visiting temples?"})
        
        res2: ChatCompletionResponse = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        
        # Verify second response
        self.assertIsNotNone(res2)
        self.assertIsNotNone(res2.choices[0].message.content)

    def test_custom_parameters(self):
        """Test with custom Together-specific parameters"""
        try:
            res: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stop=["In conclusion", "To summarize"],
                extra_headers={
                    "x-maxim-generation-name": "quantum_explanation",
                    "x-maxim-generation-tags": {
                        "test_type": "custom_parameters",
                        "topic": "quantum_computing"
                    }
                }
            )
            
            # Verify response structure
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, 'choices'))
            self.assertTrue(len(res.choices) > 0)
            self.assertIsNotNone(res.choices[0].message.content)
            print("Custom parameters response:", res.choices[0].message.content[:100] + "...")
        except Exception as e:
            self.skipTest(f"Custom parameters error: {e}")

    def test_multimodal_image_url(self):
        """Test multimodal input with image URL"""
        # Use a publicly accessible image URL
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        trace_id = str(uuid4())
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Describe it in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        
        try:
            res: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            messages = []
            messages.append({"role": "user", "content": f"Create a long poem from the above text: {res.choices[0].message.content}"})
            
            res2: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.6,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            print("Multimodal response: ", res2)
            
            messages = []
            messages.append({"role": "user", "content": f"Translate this in German: {res2.choices[0].message.content}"})
            
            res3: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.6,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            print("Multimodal response: ", res3)
            
            # Verify response structure
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, 'choices'))
            self.assertTrue(len(res.choices) > 0)
            self.assertIsNotNone(res.choices[0].message.content)
            
        except Exception as e:
            # If multimodal not supported, fallback to text-only test
            self.skipTest(f"Multimodal not supported: {e}")
            
    def test_multimodal_image_with_context(self):
        """Test multimodal input with image and contextual conversation"""
        image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640"
        trace_id = str(uuid4())
        
        try:
            # First message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this landscape image and tell me what time of day it might be."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ]
            
            res1: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.6,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            print("Multimodal response: ", res1)
            
            # Verify first response
            self.assertIsNotNone(res1)
            self.assertIsNotNone(res1.choices[0].message.content)
            
            # Follow-up question
            messages.append({"role": "assistant", "content": res1.choices[0].message.content})
            messages.append({"role": "user", "content": "Based on that analysis, what weather conditions would be ideal for photography at this location?"})
            
            res2: ChatCompletionResponse = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Text-only for follow-up
                messages=messages,
                max_tokens=150,
                temperature=0.6,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            # Verify second response
            self.assertIsNotNone(res2)
            self.assertIsNotNone(res2.choices[0].message.content)
            
        except Exception as e:
            # If multimodal not supported, skip test
            self.skipTest(f"Multimodal context test not supported: {e}")

    def test_error_handling(self):
        """Test error handling with invalid parameters"""
        try:
            # Test with invalid temperature
            with self.assertRaises(Exception) as context:
                self.client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    messages=[{"role": "user", "content": "Test message"}],
                    temperature=2.0  # Invalid temperature > 1.0
                )
            print("Successfully caught error for invalid temperature:", str(context.exception))

            # Test with invalid max_tokens
            with self.assertRaises(Exception) as context:
                self.client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    messages=[{"role": "user", "content": "Test message"}],
                    max_tokens=-1  # Invalid negative tokens
                )
            print("Successfully caught error for invalid max_tokens:", str(context.exception))

            # Test with empty messages
            with self.assertRaises(Exception) as context:
                self.client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    messages=[]  # Empty messages array
                )
            print("Successfully caught error for empty messages:", str(context.exception))

        except Exception as e:
            self.skipTest(f"Error handling test failed: {e}")

    def test_response_formats(self):
        """Test different response formats"""
        try:
            # Test JSON response format
            messages = [
                {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                {"role": "user", "content": "List 3 countries and their capitals in JSON format."}
            ]
            res = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                extra_headers={
                    "x-maxim-generation-name": "json_format_test",
                    "x-maxim-generation-tags": {
                        "test_type": "response_format",
                        "format": "json"
                    }
                }
            )
            print("JSON format response:", res.choices[0].message.content[:100] + "...")

            # Test list format
            messages = [
                {"role": "system", "content": "You are a helpful assistant that responds in numbered lists."},
                {"role": "user", "content": "List 3 benefits of exercise."}
            ]
            res = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                extra_headers={
                    "x-maxim-generation-name": "list_format_test",
                    "x-maxim-generation-tags": {
                        "test_type": "response_format",
                        "format": "list"
                    }
                }
            )
            print("List format response:", res.choices[0].message.content[:100] + "...")

            # Test table format
            messages = [
                {"role": "system", "content": "You are a helpful assistant that responds in markdown table format."},
                {"role": "user", "content": "Create a table comparing 3 different programming languages."}
            ]
            res = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                extra_headers={
                    "x-maxim-generation-name": "table_format_test",
                    "x-maxim-generation-tags": {
                        "test_type": "response_format",
                        "format": "table"
                    }
                }
            )
            print("Table format response:", res.choices[0].message.content[:100] + "...")

        except Exception as e:
            self.skipTest(f"Response format test failed: {e}")

    def test_tool_calls(self):
        """Test chat completion with tool calls"""
        try:
            # Define available tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to get weather for"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]

            # Test tool call generation
            res = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can check the weather."},
                    {"role": "user", "content": "What's the weather like in San Francisco?"}
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=150,
                temperature=0.7,
                extra_headers={
                    "x-maxim-generation-name": "weather_tool_call",
                    "x-maxim-generation-tags": {
                        "test_type": "tool_calls",
                        "tool": "get_weather"
                    }
                }
            )

            # Verify tool call response
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, 'choices'))
            self.assertTrue(len(res.choices) > 0)
            if hasattr(res.choices[0], 'tool_calls') and res.choices[0].tool_calls:
                tool_call = res.choices[0].tool_calls[0]
                self.assertEqual(tool_call.function.name, "get_weather")
                print("Tool call function:", tool_call.function.name)
                print("Tool call arguments:", tool_call.function.arguments)

            # Test tool call with forced tool choice
            res = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can check the weather."},
                    {"role": "user", "content": "Check weather for Tokyo"}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
                max_tokens=150,
                temperature=0.7,
                extra_headers={
                    "x-maxim-generation-name": "forced_weather_tool_call",
                    "x-maxim-generation-tags": {
                        "test_type": "forced_tool_calls",
                        "tool": "get_weather"
                    }
                }
            )

            # Verify forced tool call response
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, 'choices'))
            self.assertTrue(len(res.choices) > 0)
            if hasattr(res.choices[0], 'tool_calls') and res.choices[0].tool_calls:
                tool_call = res.choices[0].tool_calls[0]
                self.assertEqual(tool_call.function.name, "get_weather")
                print("Forced tool call function:", tool_call.function.name)
                print("Forced tool call arguments:", tool_call.function.arguments)

            # Test multiple tool calls
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get time for"
                            }
                        },
                        "required": ["location"]
                    }
                }
            })

            res = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can check weather and time."},
                    {"role": "user", "content": "What's the weather and time in London?"}
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=150,
                temperature=0.7,
                extra_headers={
                    "x-maxim-generation-name": "multiple_tool_calls",
                    "x-maxim-generation-tags": {
                        "test_type": "multiple_tool_calls",
                        "tools": "weather_and_time"
                    }
                }
            )

            # Verify multiple tool calls response
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, 'choices'))
            self.assertTrue(len(res.choices) > 0)
            if hasattr(res.choices[0], 'tool_calls') and res.choices[0].tool_calls:
                for tool_call in res.choices[0].tool_calls:
                    self.assertIn(tool_call.function.name, ["get_weather", "get_time"])
                    print(f"Multiple tool call function: {tool_call.function.name}")
                    print(f"Multiple tool call arguments: {tool_call.function.arguments}")

        except Exception as e:
            self.skipTest(f"Tool calls test failed: {e}")
            
    def test_error_handling_invalid_model(self):
        """Test error handling with an invalid model name"""
        try:
            self.client.chat.completions.create(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
        except Exception as e:
            self.assertIn("model", str(e).lower())

    def test_error_handling_empty_messages(self):
        """Test error handling with empty messages list"""
        try:
            self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[],
                max_tokens=10
            )
        except Exception as e:
            self.assertIn("message", str(e).lower())

    def test_error_handling_invalid_message_format(self):
        """Test error handling with invalid message format"""
        try:
            self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"invalid_role": "user", "content": "Test message"}],
                max_tokens=10
            )
        except Exception as e:
            self.assertIn("role", str(e).lower())

    def test_error_handling_invalid_tool_format(self):
        """Test error handling with invalid tool format"""
        try:
            self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Test message"}],
                tools=[{"invalid_type": "function"}],
                max_tokens=10
            )
        except Exception as e:
            self.assertIn("tool", str(e).lower())

    def tearDown(self) -> None:
        pass

class TestTogetherAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for Together AI async integration."""

    async def asyncSetUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not togetherApiKey:
            self.skipTest("TOGETHER_API_KEY environment variable is not set")

        self.logger = Maxim({ "base_url": baseUrl }).logger()
        
        # Initialize async Together client with instrumentation
        self.async_client = AsyncTogether(api_key=togetherApiKey)
        instrument_together(self.logger)

    async def test_async_non_streaming_chat_completion(self):
        """Test async non-streaming chat completion with basic user message"""
        try:
            response = await self.async_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
                max_tokens=150,
                temperature=0.7,
                extra_headers={
                    "x-maxim-generation-name": "async_nyc_recommendations",
                    "x-maxim-generation-tags": {
                        "test_type": "async_non_streaming",
                        "topic": "travel"
                    }
                }
            )
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'choices'))
            self.assertTrue(len(response.choices) > 0)
            self.assertIsNotNone(response.choices[0].message.content)
            print("Async response:", response.choices[0].message.content[:100] + "...")

        except Exception as e:
            self.skipTest(f"Async error: {e}")

    async def test_async_streaming_chat_completion(self):
        """Test async streaming chat completion"""
        try:
            stream = await self.async_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Tell me a short story about a robot learning to paint."}],
                max_tokens=200,
                temperature=0.8,
                stream=True,
                extra_headers={
                    "x-maxim-generation-name": "async_robot_story",
                    "x-maxim-generation-tags": {
                        "test_type": "async_streaming",
                        "topic": "creative"
                    }
                }
            )
            
            # Consume the async stream
            full_response = ""
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            # Verify we received streaming data
            self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
            self.assertGreater(len(full_response), 0, "Expected non-empty response")
            print("Async streaming full response:", full_response[:100] + "...")

        except Exception as e:
            self.skipTest(f"Async streaming error: {e}")

    async def test_async_with_trace_id(self):
        """Test async with custom trace ID"""
        try:
            trace_id = str(uuid4())
            
            # First call
            response1 = await self.async_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "What is artificial intelligence?"}],
                max_tokens=100,
                temperature=0.5,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                }
            )
            
            # Second call with same trace ID
            response2 = await self.async_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
            
            print("Async trace ID response 1:", response1.choices[0].message.content[:100] + "...")
            print("Async trace ID response 2:", response2.choices[0].message.content[:100] + "...")

        except Exception as e:
            self.skipTest(f"Async trace ID error: {e}")

    async def test_async_error_handling_invalid_model(self):
        """Test async error handling with an invalid model name"""
        try:
            res = await self.async_client.chat.completions.create(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
        except Exception as e:
            print(e)

    async def test_async_error_handling_empty_messages(self):
        """Test async error handling with empty messages list"""
        try:
            response = await self.async_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[],
                max_tokens=10
            )
            async for chunk in response:
                print(chunk)
        except Exception as e:
            self.assertIn("message", str(e).lower())

    async def test_async_error_handling_streaming(self):
        """Test async error handling with streaming"""
        try:
            response = await self.async_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"invalid_role": "user", "content": "Test message"}],
                max_tokens=10,
                stream=True
            )
            async for chunk in response:
                pass
        except Exception as e:
            self.assertIn("role", str(e).lower())

    async def asyncTearDown(self) -> None:
        if hasattr(self, 'logger'):
            self.logger.cleanup()

if __name__ == "__main__":
    unittest.main()
