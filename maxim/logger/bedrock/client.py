import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
from uuid import uuid4

from botocore.client import BaseClient

from ...scribe import scribe
from ..logger import Generation, GenerationConfig, Logger, Trace, TraceConfig
from .async_client import MaximBedrockAsyncClient
from .utils import BedrockUtils

# Define a type alias for message structure for clarity
BedrockMessageParam = Dict[str, Union[str, List[Dict[str, str]]]]
BedrockStreamEvent = Dict[str, Any]
BedrockMessage = Dict[str, Any]

class MaximBedrockClient:
    def __init__(
        self,
        logger: Logger,
        client: BaseClient,
    ):
        self._client = client
        self._logger = logger
        self._async_client = MaximBedrockAsyncClient(logger, client)


    def converse_stream(
        self,
        messages: Iterable[BedrockMessageParam],
        system: Optional[Union[str, List[Dict[str, str]]]] = None,
        *,
        modelId: str,
        max_tokens: Optional[int] = None,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
        inferenceConfig: Optional[Dict[str, Any]] = None,
        toolConfig: Optional[Dict[str, Any]] = None,
        guardrailConfig: Optional[Dict[str, Any]] = None,
        performanceConfig: Optional[Dict[str, Any]] = None,
        additionalModelRequestFields: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[BedrockStreamEvent]:
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        model = BedrockUtils.get_model_name(modelId)

        final_inference_config = {
            **(inferenceConfig or {}),
            **({ 'maxTokens': max_tokens } if max_tokens is not None else {}),
            **kwargs.get('inference_config', {})
        }
        final_inference_config = {k: v for k, v in final_inference_config.items() if v is not None}    
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            generation_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="bedrock",
                name=generation_name,
                model_parameters={
                    "model_id": modelId,
                    **final_inference_config,
                    **(additionalModelRequestFields or {}),
                },
                messages=BedrockUtils.parse_message_param(messages),
            )
            generation = trace.generation(generation_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][BedrockClient] Error preparing logging: {str(e)}"
            )

        api_params = {
            "modelId": modelId,
            "messages": list(messages),
        }
        if system:
            api_params["system"] = [{"text": system}] if isinstance(system, str) else system
        if final_inference_config:
            api_params["inferenceConfig"] = final_inference_config
        if toolConfig:
            api_params["toolConfig"] = toolConfig
        if guardrailConfig:
            api_params["guardrailConfig"] = guardrailConfig
        if performanceConfig:
            api_params["performanceConfig"] = performanceConfig
        if additionalModelRequestFields:
             api_params["additionalModelRequestFields"] = additionalModelRequestFields

        response_stream = self._client.converse_stream(**api_params)
        
        try:
            if generation is not None:
                with response_stream as stream:
                    # Initialize combined response object
                    combined_response = {
                        "id": str(uuid4()),
                        "created": int(time.time()),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": ""
                            },
                            "finish_reason": None
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    
                    for event in stream:
                        event_type = event.get("event-type")
                        
                        if event_type == "contentBlockDelta":
                            delta = event.get("delta", {})
                            if "text" in delta:
                                # Append to the combined content
                                combined_response["choices"][0]["message"]["content"] += delta["text"]
                                
                        elif event_type == "messageStop":
                            # Set finish reason
                            combined_response["choices"][0]["finish_reason"] = event.get("stopReason", "stop")
                            
                        elif event_type == "metadata":
                            usage = event.get("usage", {})
                            if usage:
                                # Update usage information
                                combined_response["usage"] = {
                                    "prompt_tokens": usage.get("inputTokens", 0),
                                    "completion_tokens": usage.get("outputTokens", 0),
                                    "total_tokens": usage.get("totalTokens", 0)
                                }
                            else:
                                scribe().warning(
                                    "[MaximSDK][BedrockClient] No usage metrics available in metadata"
                                )
                        
                        # Pass through the event for streaming
                        yield event
                    
                    # After collecting all events, send the combined response to generation
                    if generation is not None:
                        generation.result(combined_response)
                        if is_local_trace and trace is not None:
                            trace.set_output(combined_response["choices"][0]["message"]["content"])
                            trace.end()

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][BedrockClient] Error processing Bedrock stream or logging: {str(e)}"
            )


    def converse(
        self,
        messages: Iterable[BedrockMessageParam],
        system: Optional[Union[str, List[Dict[str, str]]]] = None,
        *,
        modelId: str,
        max_tokens: Optional[int] = None,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
        inferenceConfig: Optional[Dict[str, Any]] = None,
        toolConfig: Optional[Dict[str, Any]] = None,
        guardrailConfig: Optional[Dict[str, Any]] = None,
        performanceConfig: Optional[Dict[str, Any]] = None,
        additionalModelRequestFields: Optional[Dict[str, Any]] = None,
         **kwargs: Any,
    ) -> BedrockMessage:
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        model = BedrockUtils.get_model_name(modelId)
        final_inference_config = {
            **(inferenceConfig or {}),
            **({ 'maxTokens': max_tokens } if max_tokens is not None else {}),
            **kwargs.get('inference_config', {})
        }
        final_inference_config = {k: v for k, v in final_inference_config.items() if v is not None}
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            generation_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="bedrock",
                name=generation_name,
                model_parameters={
                    "model_id": modelId,
                    **final_inference_config,
                     **(additionalModelRequestFields or {}),
                },
                messages=BedrockUtils.parse_message_param(messages),
            )
            generation = trace.generation(generation_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][BedrockClient] Error preparing logging: {str(e)}"
            )
        api_params = {
            "modelId": modelId,
            "messages": list(messages),
        }
        if system:
            api_params["system"] = [{"text": system}] if isinstance(system, str) else system
        if final_inference_config:
            api_params["inferenceConfig"] = final_inference_config
        if toolConfig:
            api_params["toolConfig"] = toolConfig
        if guardrailConfig:
            api_params["guardrailConfig"] = guardrailConfig
        if performanceConfig:
            api_params["performanceConfig"] = performanceConfig
        if additionalModelRequestFields:
             api_params["additionalModelRequestFields"] = additionalModelRequestFields        
        response = self._client.converse(**api_params)
        changed_response = BedrockUtils.parse_message(response)        
        try:
            if generation is not None:
                generation.result(changed_response)
                if is_local_trace and trace is not None:
                    if changed_response is not None:
                        content = changed_response.get("choices", [])[0].get("message", {}).get("content", "")
                        trace.set_output(content)
                    generation.end()
                    trace.end()

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][BedrockClient] Error processing Bedrock stream or logging: {str(e)}"
            )
        return response
                    
    @property
    def async_client(self) -> MaximBedrockAsyncClient:
        return self._async_client
