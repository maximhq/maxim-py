import os
import time
from uuid import uuid4
from dotenv import load_dotenv

from maxim.maxim import Maxim

import logging

logging.basicConfig(level=logging.DEBUG)


# Load environment variables from .env file
load_dotenv(override=True)

maxim = Maxim(
    {
        "api_key": os.getenv("MAXIM_API_KEY"),
        "base_url": os.getenv("MAXIM_BASE_URL"),
        "debug": True,
    }
)


# Initializing logger
logger = maxim.logger({"id": os.getenv("MAXIM_LOG_REPO_ID")})

print(os.getenv("MAXIM_LOG_REPO_ID"))

for i in range(10):

    trace = logger.trace({"id": str(uuid4()), "name": "pre-defined-trace"})

    generation = trace.generation(
        {
            "id": str(uuid4()),
            "name": "customer-support--gather-information",
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "you are an helpful assistant who helps gather customer information",
                },
                {"role": "user", "content": "My internet is not working"},
            ],
            "model_parameters": {"temperature": 0.7},
        }
    )

    generation.result(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Lol ¯\_(ツ)_/¯ "
                        * (2 * 1024 * 1024 // len("Lol ¯\_(ツ)_/¯ ")),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
    )

    trace.end()
logger.flush()
