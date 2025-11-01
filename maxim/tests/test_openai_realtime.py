import os
import asyncio
from dotenv import load_dotenv
# import logging

from openai import OpenAI
from maxim import Maxim
from maxim.logger.openai import MaximOpenAIClient

load_dotenv()

# Azure OpenAI Realtime Docs

# How-to: https://learn.microsoft.com/azure/ai-services/openai/how-to/realtime-audio
# Supported models and API versions: https://learn.microsoft.com/azure/ai-services/openai/how-to/realtime-audio#supported-models
# Entra ID auth: https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity

# logging.basicConfig(level=logging.DEBUG)
openaiapikey = os.getenv("OPENAI_API_KEY")
apikey = os.getenv("MAXIM_API_KEY")
baseURL = os.getenv("MAXIM_BASE_URL")


async def main() -> None:
    """The following example demonstrates how to configure OpenAI to use the Realtime API.
    For an audio example, see push_to_talk_app.py and update the client and model parameter accordingly.

    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """
    logger = Maxim({"base_url": baseURL}).logger()

    client = MaximOpenAIClient(OpenAI(api_key=openaiapikey), logger=logger).aio
    async with client.realtime.connect(
        model="gpt-realtime",
    ) as connection:
        await connection.session.update(
            session={
                "output_modalities": ["text"],
                "model": "gpt-realtime",
                "type": "realtime",
            }
        )
        while True:
            user_input = input("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await connection.response.create()
            async for event in connection:
                if event.type == "response.output_text.delta":
                    print(event.delta, flush=True, end="")
                elif event.type == "response.output_text.done":
                    print()
                elif event.type == "response.done":
                    break


asyncio.run(main())
