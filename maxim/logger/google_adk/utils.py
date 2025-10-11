"""Utility functions for Google ADK Maxim integration."""

import uuid
from typing import Any, Dict, List


def dictify(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary representation."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    return {"value": str(obj)}


def google_adk_postprocess_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process inputs for Google ADK operations."""
    processed = {}
    
    # Extract key information from inputs
    if "self" in inputs:
        obj = inputs["self"]
        if hasattr(obj, "name"):
            processed["name"] = obj.name
        if hasattr(obj, "description"):
            processed["description"] = obj.description
        if hasattr(obj, "model"):
            processed["model"] = str(obj.model)
    
    # Extract arguments
    if "args" in inputs:
        processed["args"] = inputs["args"]
    if "kwargs" in inputs:
        processed["kwargs"] = inputs["kwargs"]
    
    # Extract new_message if present
    if "new_message" in inputs:
        new_message = inputs["new_message"]
        if hasattr(new_message, "parts"):
            processed["message_parts"] = len(new_message.parts)
            if new_message.parts:
                processed["message_text"] = str(new_message.parts[0].text) if hasattr(new_message.parts[0], "text") else ""
    
    return processed


def extract_tool_details(tool) -> Dict[str, Any]:
    """Extract tool details from a tool object."""
    details = {
        "name": getattr(tool, "name", "unknown"),
        "description": getattr(tool, "description", "unknown"),
        "args": None,
    }
    
    # Try to extract function signature if available
    if hasattr(tool, "func"):
        try:
            import inspect
            sig = inspect.signature(tool.func)
            details["args"] = {
                param.name: str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                for param in sig.parameters.values()
            }
        except Exception:
            pass
    
    return details


def get_agent_display_name(inputs: Dict[str, Any]) -> str:
    """Get display name for agent operations."""
    if "self" in inputs:
        obj = inputs["self"]
        if hasattr(obj, "name"):
            return f"Agent: {obj.name}"
        if hasattr(obj, "role"):
            return f"Agent: {obj.role}"
    return "Agent Operation"


def get_tool_display_name(inputs: Dict[str, Any]) -> str:
    """Get display name for tool operations."""
    if "self" in inputs:
        obj = inputs["self"]
        if hasattr(obj, "name"):
            return f"Tool: {obj.name}"
    return "Tool Operation"


def extract_usage_from_response(response) -> Dict[str, int]:
    """Extract usage information from various response types."""
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    try:
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                usage_info["prompt_tokens"] = getattr(usage, 'prompt_token_count', 0)
            if hasattr(usage, 'candidates_token_count'):
                usage_info["completion_tokens"] = getattr(usage, 'candidates_token_count', 0)
            if hasattr(usage, 'total_token_count'):
                usage_info["total_tokens"] = getattr(usage, 'total_token_count', 0)
            elif usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
        elif hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, 'prompt_tokens'):
                usage_info["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0)
            if hasattr(usage, 'completion_tokens'):
                usage_info["completion_tokens"] = getattr(usage, 'completion_tokens', 0)
            if hasattr(usage, 'total_tokens'):
                usage_info["total_tokens"] = getattr(usage, 'total_tokens', 0)
    except Exception as e:
        from ...scribe import scribe
        scribe().debug(f"[MaximSDK] Error extracting usage: {e}")
    
    return usage_info


def extract_model_info(model_obj) -> Dict[str, str]:
    """Extract model information from model object."""
    model_info = {"model": "unknown", "provider": "google"}
    
    try:
        if hasattr(model_obj, 'model'):
            model_info["model"] = str(model_obj.model)
        elif hasattr(model_obj, 'model_name'):
            model_info["model"] = model_obj.model_name or "unknown"
        
        # Determine provider based on model name
        model_name = model_info["model"].lower()
        if "gemini" in model_name:
            model_info["provider"] = "google"
        elif "gpt" in model_name:
            model_info["provider"] = "openai"
        elif "claude" in model_name:
            model_info["provider"] = "anthropic"
    except Exception as e:
        from ...scribe import scribe
        scribe().debug(f"[MaximSDK] Error extracting model info: {e}")
    
    return model_info


def convert_messages_to_maxim_format(contents) -> List[Dict[str, Any]]:
    """Convert Google ADK contents to Maxim format."""
    messages = []
    try:
        for content in contents:
            if hasattr(content, 'parts'):
                maxim_msg = {
                    "role": getattr(content, 'role', 'user'),
                    "content": []
                }
                
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        maxim_msg["content"].append({
                            "type": "text",
                            "text": str(part.text)
                        })
                    elif hasattr(part, 'function_call'):
                        # Handle function calls
                        function_call = part.function_call
                        maxim_msg["content"].append({
                            "type": "function_call",
                            "function_call": {
                                "name": getattr(function_call, 'name', 'unknown'),
                                "arguments": str(getattr(function_call, 'args', {}))
                            }
                        })
                    elif hasattr(part, 'function_response'):
                        # Handle function responses
                        function_response = part.function_response
                        maxim_msg["content"].append({
                            "type": "function_response",
                            "function_response": {
                                "name": getattr(function_response, 'name', 'unknown'),
                                "response": str(getattr(function_response, 'response', ''))
                            }
                        })
                
                if not maxim_msg["content"]:
                    maxim_msg["content"] = str(content)
                
                messages.append(maxim_msg)
            else:
                # Fallback for other content types
                messages.append({
                    "role": "user",
                    "content": str(content)
                })
    except Exception as e:
        from ...scribe import scribe
        scribe().debug(f"[MaximSDK] Error converting messages: {e}")
        # Fallback to simple string conversion
        messages = [str(content) for content in contents]
    
    return messages


def extract_content_from_response(response) -> str:
    """Extract meaningful content from Google ADK response."""
    try:
        if hasattr(response, 'content') and response.content:
            if hasattr(response.content, 'parts'):
                content_parts = []
                for part in response.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content_parts.append(part.text)
                    elif hasattr(part, 'function_call'):
                        # For function calls, show the function name and args
                        function_call = part.function_call
                        function_info = f"Function: {getattr(function_call, 'name', 'unknown')}"
                        if hasattr(function_call, 'args'):
                            function_info += f"({function_call.args})"
                        content_parts.append(function_info)
                return " | ".join(content_parts) if content_parts else str(response)
        return str(response)
    except Exception as e:
        from ...scribe import scribe
        scribe().debug(f"[MaximSDK] Error extracting content: {e}")
        return str(response)


def extract_tool_calls_from_response(response) -> List[Dict[str, Any]]:
    """Extract tool calls from Google ADK response."""
    tool_calls = []
    
    try:
        from ...scribe import scribe
        scribe().info(f"[MaximSDK] Response type: {type(response)}")
        scribe().info(f"[MaximSDK] Has content: {hasattr(response, 'content')}")
        
        # Check if response has tool calls in content parts
        if hasattr(response, 'content') and response.content and hasattr(response.content, 'parts'):
            scribe().info(f"[MaximSDK] Number of parts: {len(response.content.parts)}")
            for idx, part in enumerate(response.content.parts):
                scribe().info(f"[MaximSDK] Part {idx} type: {type(part)}, has function_call: {hasattr(part, 'function_call')}")
                # Check if function_call exists AND is not None
                if hasattr(part, 'function_call') and part.function_call is not None:
                    function_call = part.function_call
                    scribe().info(f"[MaximSDK] Function call: {function_call}")
                    scribe().info(f"[MaximSDK] Function call name: {getattr(function_call, 'name', 'NOT_FOUND')}")
                    
                    # Generate a unique ID for the tool call using function name and index
                    func_name = getattr(function_call, 'name', 'unknown')
                    tool_call_id = f"{func_name}_{idx}_{str(uuid.uuid4())[:8]}"
                    
                    tool_call_data = {
                        "name": func_name,
                        "args": dict(getattr(function_call, 'args', {})) if hasattr(function_call, 'args') else {},
                        "tool_call_id": tool_call_id
                    }
                    tool_calls.append(tool_call_data)
                    
                    scribe().info(f"[MaximSDK] Extracted tool call: {tool_call_data['name']} with args: {tool_call_data['args']}")
        
        # Check if response has tool calls directly
        if hasattr(response, 'tool_calls'):
            for idx, tool_call in enumerate(response.tool_calls):
                tool_call_id = f"{getattr(tool_call, 'name', 'unknown')}_{idx}_{str(uuid.uuid4())[:8]}"
                tool_call_data = {
                    "name": getattr(tool_call, 'name', 'unknown'),
                    "args": dict(getattr(tool_call, 'args', {})) if hasattr(tool_call, 'args') else {},
                    "tool_call_id": tool_call_id
                }
                tool_calls.append(tool_call_data)
                
    except Exception as e:
        from ...scribe import scribe
        scribe().error(f"[MaximSDK] Error extracting tool calls: {e}")
        import traceback
        traceback.print_exc()
    
    return tool_calls
