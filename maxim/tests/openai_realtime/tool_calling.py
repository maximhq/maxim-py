import os
import json
import asyncio
from dotenv import load_dotenv

from openai import OpenAI
from maxim import Maxim
from maxim.logger.openai import MaximOpenAIClient

load_dotenv()

openaiapikey = os.getenv("OPENAI_API_KEY")
apikey = os.getenv("MAXIM_API_KEY")
baseURL = os.getenv("MAXIM_BASE_URL")

# Example tool (function)
def get_weather(location: str) -> str:
    # Mock implementation — replace with real API if needed
    weather_data = {
        "London": "Cloudy, 12°C",
        "New York": "Sunny, 18°C",
        "Bengaluru": "Partly cloudy, 26°C",
    }
    return weather_data.get(location, f"Sorry, I don't have data for {location}.")

async def main():
    logger = Maxim({"base_url": baseURL}).logger()
    client = MaximOpenAIClient(OpenAI(api_key=openaiapikey), logger=logger).aio

    async with client.realtime.connect(model="gpt-realtime") as connection:
        # --- Define available tools for the model ---
        await connection.session.update(
            session={
                "model": "gpt-realtime",
                "type": "realtime",
                "output_modalities": ["text"],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the current weather for a given location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                            },
                            "required": ["location"],
                        },
                    }
                ],
                "tool_choice": "auto",  # Let the model decide when to use a tool
            }
        )

        print("Realtime session started. Type a message, 'q' to quit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "q":
                break

            # Send user input
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )

            await connection.response.create()

            # Flag to track if we need to continue listening after tool execution
            tool_executed = False

            async for event in connection:
                if event.type == "response.output_text.delta":
                    print(event.delta, flush=True, end="")

                elif event.type == "response.output_text.done":
                    print()

                # 🧠 Detect if the model is calling a tool
                elif event.type == "response.function_call_arguments.delta":
                    # The model is sending function call arguments (partial stream)
                    print(f"[tool call args delta]: {event.delta}")

                elif event.type == "response.function_call_arguments.done":
                    fn_name = event.name
                    args_string = event.arguments
                    args = json.loads(args_string) if isinstance(args_string, str) else args_string

                    print(f"\nTool call complete: {fn_name}({args})")
                    if fn_name == "get_weather":
                        location = args.get("location")
                        result = get_weather(location)
                        print(f"[Tool executed] => {result}")

                        # 1️⃣ Add the tool output as a conversation item
                        await connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "tool",
                                "name": fn_name,
                                "content": [{"type": "output_text", "text": result}],
                            }
                        )

                        # Set flag to continue listening
                        tool_executed = True

                elif event.type == "response.done":
                    # If a tool was executed, request another response and continue
                    if tool_executed:
                        print("\n[Waiting for assistant response...]")
                        await connection.response.create()
                        tool_executed = False
                        continue
                    else:
                        # Normal conversation turn is complete
                        break

asyncio.run(main())