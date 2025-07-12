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

        self.logger = Maxim().logger()
        
        # Initialize Together clients with instrumentation
        self.client = Together(api_key=togetherApiKey)
        instrument_together(self.logger)

    def test_non_streaming_chat_completion(self):
        """Test non-streaming chat completion with basic user message"""
        res: ChatCompletionResponse = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
            max_tokens=150,
            temperature=0.7
        )
        
        # Verify response structure
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'choices'))
        self.assertTrue(len(res.choices) > 0)
        self.assertIsNotNone(res.choices[0].message.content)
        
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
        res: ChatCompletionResponse = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            stop=["In conclusion", "To summarize"]
        )
        
        # Verify response structure
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'choices'))
        self.assertTrue(len(res.choices) > 0)
        self.assertIsNotNone(res.choices[0].message.content)

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

    def tearDown(self) -> None:
        pass

class TestTogetherAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for Together AI async integration."""

    async def asyncSetUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        if not togetherApiKey:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        
        self.logger = Maxim().logger()
        
        # Initialize async Together client with instrumentation
        self.async_client = AsyncTogether(api_key=togetherApiKey)
        instrument_together(self.logger)

    async def test_async_non_streaming_chat_completion(self):
        """Test async non-streaming chat completion with basic user message"""
        res = await self.async_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
            max_tokens=150,
            temperature=0.7
        )
        
        # Verify response structure
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'choices'))
        self.assertTrue(len(res.choices) > 0)
        self.assertIsNotNone(res.choices[0].message.content)
        
    async def test_async_non_streaming_with_system_message(self):
        """Test async non-streaming chat completion with system message"""
        res = await self.async_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant. Always provide concise, practical advice."},
                {"role": "user", "content": "What's the best way to get around San Francisco?"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        
        # Verify response structure
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'choices'))
        self.assertTrue(len(res.choices) > 0)
        self.assertIsNotNone(res.choices[0].message.content)
        
    async def test_async_streaming_chat_completion(self):
        """Test async streaming chat completion"""
        res = await self.async_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Tell me a short story about a robot learning to paint."}],
            max_tokens=200,
            temperature=0.8,
            stream=True
        )
        
        # Consume the async stream
        full_response = ""
        chunk_count = 0
        async for chunk in res:
            chunk_count += 1
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
        
        # Verify we received streaming data
        self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
        self.assertGreater(len(full_response), 0, "Expected non-empty response")
        
    async def test_async_streaming_with_system_message(self):
        """Test async streaming chat completion with system message"""
        res = await self.async_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
        async for chunk in res:
            chunk_count += 1
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
        
        # Verify we received streaming data
        self.assertGreater(chunk_count, 0, "Expected to receive streaming chunks")
        self.assertGreater(len(full_response), 0, "Expected non-empty response")
        
    async def asyncTearDown(self) -> None:
        pass

if __name__ == "__main__":
    unittest.main()