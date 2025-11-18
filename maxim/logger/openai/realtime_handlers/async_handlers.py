from openai.types.realtime import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
)


def handle_text_response(
    response: RealtimeConversationItemSystemMessage
    | RealtimeConversationItemUserMessage
    | RealtimeConversationItemAssistantMessage,
) -> str:
    if len(response.content) > 0:
        if response.content[0].type == "output_text":
            return response.content[0].text
        elif response.content[0].type == "output_audio":
            return response.content[0].transcript
    return ""


def handle_function_call_response(
    response: RealtimeConversationItemFunctionCall,
) -> dict[str, str]:
    return {
        "name": response.name,
        "arguments": response.arguments,
    }


def handle_conversation_item_message(
    item: RealtimeConversationItemSystemMessage
    | RealtimeConversationItemUserMessage
    | RealtimeConversationItemAssistantMessage,
) -> str:
    content = item.content

    # Parse content to text - handle Pydantic models
    message_text: str = ""
    if isinstance(content, list):
        for content_item in content:
            # Handle Pydantic Content model
            if hasattr(content_item, "type"):
                content_type = content_item.type
                if content_type == "input_text":
                    text = content_item.text
                    if isinstance(text, str):
                        message_text += text
                elif content_type == "input_image":
                    message_text += "[image]"
                elif content_type == "input_file":
                    message_text += "[file]"
                elif content_type == "input_audio":
                    # For audio, use transcript if available
                    transcript = getattr(content_item, "transcript", None)
                    if isinstance(transcript, str):
                        message_text += transcript
                    else:
                        message_text += "[audio]"
            elif isinstance(content_item, dict):
                content_type = content_item.get("type")
                if content_type == "input_text":
                    text = content_item.get("text", "")
                    if isinstance(text, str):
                        message_text += text
                elif content_type == "input_image":
                    message_text += "[image]"
                elif content_type == "input_file":
                    message_text += "[file]"
                elif content_type == "input_audio":
                    transcript = content_item.get("transcript")
                    if isinstance(transcript, str):
                        message_text += transcript
                    else:
                        message_text += "[audio]"
            elif isinstance(content_item, str):
                message_text += content_item
    elif isinstance(content, str):
        message_text = content

    return message_text


# TODO: Might not be needed
def handle_conversation_item_function_call(
    item: RealtimeConversationItemFunctionCall,
) -> dict[str, str]:
    return {
        "name": item.name,
        "arguments": item.arguments,
    }
