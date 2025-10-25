import os
import unittest
from uuid import uuid4

import dotenv
from openai import OpenAI
from maxim.logger.openai import MaximOpenAIClient

from maxim import Maxim


dotenv.load_dotenv()


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestOpenAIResponsesStreamingLogger(unittest.TestCase):
    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim(
            {
                "api_key": os.getenv("MAXIM_API_KEY"),
                "base_url": os.getenv("MAXIM_BASE_URL"),
                "debug": True,
            }
        )
        self.logger = self.maxim.logger({"id": str(os.getenv("MAXIM_LOG_REPO_ID"))})
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def tearDown(self):
        if self.logger:
            self.logger.flush()

    def test_streaming_responses_logging(self):
        # Create local trace/generation via headers
        headers = {
            "x-maxim-generation-name": "Responses API Streaming Test",
        }

        wrapper = MaximOpenAIClient(self.client, logger=self.logger)
        with wrapper.responses.stream(
            model="gpt-4o",
            input="Write a 1-2 sentence haiku about autumn.",
            extra_headers=headers,
        ) as stream:
            # Iterate events and accumulate text to ensure stream behaves
            chunks = 0
            text_accum = []
            for event in stream:
                print(f"Event: {event}")
                chunks += 1
                t = getattr(event, "delta", None)
                if isinstance(t, str) and t:
                    text_accum.append(t)

            # Ensure we saw at least one event
            self.assertGreater(chunks, 0)
            # Combined text should be non-empty if deltas were present
            combined = "".join(text_accum)
            print(f"\n===\n{combined}\n===\n")
            self.assertIsInstance(combined, str)

    def test_non_streaming_responses_logging(self):
        # Create local trace/generation via headers
        headers = {
            "x-maxim-generation-name": "Responses API Non-Streaming Test",
        }

        wrapper = MaximOpenAIClient(self.client, logger=self.logger)
        response = wrapper.responses.create(
            model="gpt-4o",
            input="Explain JSON in one sentence.",
            extra_headers=headers,
        )

        # Basic assertions on Responses API object
        self.assertIsNotNone(getattr(response, "id", None))
        status = getattr(response, "status", None)
        self.assertIsInstance(status, str)
        # output_text should exist for basic text generations
        output_text = getattr(response, "output_text", None)
        self.assertIsInstance(output_text, str)


if __name__ == "__main__":
    unittest.main()
