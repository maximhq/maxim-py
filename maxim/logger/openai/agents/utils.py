import json
from typing import Optional

from openai.types.responses import Response, ResponseInputItemParam

from ....logger import (
    GenerationRequestMessage,
    GenerationResult,
    GenerationResultChoice,
    GenerationUsage,
)
from ....scribe import scribe


def parse_response_output(response:Response)-> Optional[GenerationResult]:
    try:
        usage = GenerationUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
        if response.usage is not None:
            usage["prompt_tokens"] = response.usage.input_tokens
            usage["completion_tokens"] = response.usage.output_tokens
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        choices:list[GenerationResultChoice] = []
        index =0
        for output in response.output:
            if output.type == "message":
                if output.content is None:
                    continue
                final_content = ""
                for content in output.content:
                    if content.type == "output_text":
                        final_content = content.text
                    elif content.type == "refusal":
                        final_content = content.refusal
                choices.append({
                    "index":index,
                    "message":{
                        "role":output.role,
                        "content": final_content,
                        "tool_calls": None,
                    },
                    "finish_reason":"stop",
                    "logprobs":None
                })
            elif output.type == "function_call":
                choices.append({
                    "index":index,
                    "message":{
                        "role":"assistant",
                        "content": None,
                        "tool_calls": [{
                            "type":output.type,
                            "id": output.call_id,
                            "function":{
                                "name":output.name,
                                "arguments":output.arguments
                            }
                        }],
                    },
                    "finish_reason":"stop",
                    "logprobs":None
                })
            index +=1
        return {
            "id": response.id,
            "created":  int(response.created_at),
            "choices": choices,
            "usage": usage,
            "object": "chat.completion",
            "model": response.model
        }
    except Exception as e:
        scribe().error(f"[MaximSDK] Error while parsing response output {str(e)}")        
        return None

def parse_response_input(
    input_items: list[ResponseInputItemParam],
) -> list[GenerationRequestMessage]:
    result = []
    # Handle empty list case
    if not input_items:
        return result
    # Process each item in the input list
    for input_item in input_items:
        role: str = "user"
        content: str = ""
        # Case 1: Dictionary-like with 'role' and 'content' keys
        if isinstance(input_item, dict):
            # Extract role if available
            if "role" in input_item and isinstance(input_item["role"], str):
                role = input_item["role"]
            # Extract content based on different formats
            if "content" in input_item:
                if isinstance(input_item["content"], str):
                    content = input_item["content"]
                elif isinstance(input_item["content"], list):
                    # Handle content that might be a list of content parts
                    content_parts = []
                    for part in input_item["content"]:
                        if isinstance(part, str):
                            content_parts.append(part)
                        elif isinstance(part, dict) and "text" in part:
                            content_parts.append(part["text"])
                        elif isinstance(part, dict) and "content" in part:
                            content_parts.append(part["content"])
                    content = "\n".join(content_parts)
                # Add the processed message to the result list
                result.append(GenerationRequestMessage(role=role, content=content))
            elif "type" in input_item:
                if input_item["type"] == "function_call":
                    role = "Tool"
                    content = json.dumps(
                        {
                            "id": input_item["id"] if "id" in input_item else None,
                            "call_id": (
                                input_item["call_id"]
                                if "call_id" in input_item
                                else None
                            ),
                            "name": (
                                input_item["name"] if "name" in input_item else None
                            ),
                            "arguments": (
                                input_item["arguments"]
                                if "arguments" in input_item
                                else None
                            ),
                        }
                    )
                    # Add the processed message to the result list
                    result.append(GenerationRequestMessage(role=role, content=content))
    return result

def parse_transcription_output(output:str, response_id:str, created_at:int)-> Optional[GenerationResult]:
    try:
        return {
            "id": response_id,
            "created":  created_at,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output,
                },
                "finish_reason": "stop",
                "logprobs": None
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "object": "chat.completion",
            "model": "unknown"
        }
    except Exception as e:
        scribe().error(f"[MaximSDK] Error while parsing transcription output {str(e)}")        
        return None