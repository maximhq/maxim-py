import json
import logging
from typing import Any, Dict, Optional

from agents import AgentSpanData, HandoffSpanData, guardrail
from agents.tracing import Span, Trace, TracingProcessor
from agents.tracing.create import FunctionSpanData, GenerationSpanData, GuardrailSpanData
from agents.tracing.span_data import ResponseSpanData

from ...logger import GenerationConfig, Logger, SpanConfig, GenerationError,ToolCallConfig
from ...models import Container, SpanContainer, TraceContainer
from .utils import parse_response_input, parse_response_output


class MaximOpenAIAgentsTracingProcessor(TracingProcessor):

    def __init__(self, logger: Logger):
        self.logger = logger
        self.containers: Dict[str, Container] = {}

    def __get_container(
        self, id: str, parent_id: Optional[str] = None
    ) -> Optional[Container]:
        if parent_id is not None:
            # This is the first activity in this run
            return self.containers.get(parent_id)
        else:
            return self.containers.get(str(id))
        return None

    def on_trace_start(self, trace: Trace) -> None:
        session_id = None
        if (exported_trace := trace.export()) is not None and exported_trace.get("group_id",None) is not None:
            session_id = exported_trace.get("group_id",None)
        container = TraceContainer(
            self.logger,
            trace_id=trace.trace_id,
            parent=session_id,
        )
        if container is None:
            logging.info(
                "[MaximSDK] Couldn't log trace start as container is empty. Please raise an issue with Maxim engineering team if this is unexpected."
            )
            return
        self.containers[trace.trace_id] = container
        container.set_name(trace.name)

        if not container.is_created():
            container.create()

    def on_span_start(self, span: Span[Any]) -> None:
        container: Optional[Container] = self.__get_container(
            span.trace_id, span.parent_id
        )
        if container is None:
            logging.info(
                "[MaximSDK] Couldn't log span start as container is empty. Please raise an issue with Maxim engineering team if this is unexpected."
            )
            return
        if not container.is_created():
            container.create()
        # Checking other cases
        if isinstance(span.span_data, AgentSpanData):
            span_config = SpanConfig(id=span.span_id)
            agent_span_data: AgentSpanData = span.span_data
            span_config.name = agent_span_data.name
            container.add_span(span_config)
            span_container = SpanContainer(
                span_id=span.span_id, logger=self.logger, mark_created=True
            )
            span_container.add_tags({"oaa_type": "agent"})
            span_container.add_metadata(
                {
                    "handoffs": json.dumps(agent_span_data.handoffs or []),
                    "available_tools": json.dumps(agent_span_data.tools),
                    "output_type": agent_span_data.output_type or "unknown",
                }
            )
            self.containers[span.span_id] = span_container
            return
        if isinstance(span.span_data, ResponseSpanData):
            generation_config = GenerationConfig(
                id=span.span_id, provider="openai", model="unknown"
            )
            generation_config.name = "POST /response"
            generation = container.add_generation(generation_config)
            generation.add_tag("oaa_type", "response")
            return
        if isinstance(span.span_data, FunctionSpanData):
            function_span_data:FunctionSpanData = span.span_data
            tool_config = ToolCallConfig(id=span.span_id,name=function_span_data.name,description="",args=function_span_data.input or "")
            tool = container.add_tool_call(tool_config)
            tool.add_tag("oaa_type", "function")
            return
        if isinstance(span.span_data,GenerationSpanData):
            generation_data:GenerationSpanData = span.span_data
            generation_config = GenerationConfig(id=span.span_id,provider="openai",model=generation_data.model or "unknown")
            if generation_data.model_config:
                for key, value in generation_data.model_config.items():
                    generation_config.model_parameters[key] = value
            generation = container.add_generation(generation_config)
            generation.add_tag("oaa_type", "generation")
            return
        logging.info(f"[MaximSDK] Invalid span type {span.span_data.type}")

    def on_span_end(self, span: Span[Any]) -> None:
        container: Optional[Container] = self.__get_container(span.span_id)
        if container is None:
            if isinstance(span.span_data, FunctionSpanData):
                self.logger.tool_call_update(span.span_id,{"args": span.span_data.input or ""})
                self.logger.tool_call_result(span.span_id,span.span_data.output)
                return
            if isinstance(span.span_data, GuardrailSpanData):
                guradrail_data:GuardrailSpanData = span.span_data
                container = self.__get_container(span.trace_id, span.parent_id)
                if container is not None:
                    container.add_event(
                        span.span_id,
                        f"Guardrail - {guradrail_data.name}",
                        {"oaa_type": "guardrail"},
                        {"trigger": guradrail_data.triggered})
                return
            if isinstance(span.span_data, GenerationSpanData):
                try:
                    self.logger.generation_result(span.span_id, span.span_data.output.__dict__)
                except Exception as e:
                    logging.error(f"[MaximSDK] Could not parse generation output: {str(e)}")
            if isinstance(span.span_data, ResponseSpanData):
                if (generation_input := span.span_data.input) is not None:
                    if isinstance(generation_input, str):
                        self.logger.generation_add_message(
                            span.span_id, {"role": "user", "content": generation_input}
                        )
                    if isinstance(generation_input, list):
                        messages = parse_response_input(generation_input)
                        for message in messages:
                            self.logger.generation_add_message(span.span_id, message)
                        # as a safe side we attach this entire thing in metadata
                        self.logger.generation_add_metadata(
                            span.span_id, {"Raw inputs": json.dumps(generation_input)}
                        )
                if (response := span.span_data.response) is not None:
                    self.logger.generation_set_model(span.span_id,response.model)
                    self.logger.generation_add_message(span.span_id,
                        {"role": "system", "content": response.instructions or ""}
                    )
                    self.logger.generation_set_model_parameters(span.span_id,
                        {
                            "temperature": response.temperature,
                            "tool_choice": response.tool_choice,
                            "top_p": response.top_p,
                            "max_output_tokens": response.max_output_tokens,
                            "truncation": response.truncation,
                            "parallel_tool_calls": response.parallel_tool_calls,
                            "tools": [tool.to_dict() for tool in response.tools] if response.tools else None,
                        }
                    )
                    # Checking if the response errored out
                    if response.error is not None:
                        self.logger.generation_error(span.span_id,GenerationError(message=response.error.message,code=response.error.code))
                    else:
                        result = parse_response_output(response)
                        if result is not None:
                            self.logger.generation_result(span.span_id, result)
                    self.logger.generation_add_metadata(
                        span.span_id, {"Raw response": response.model_dump_json()}
                    )
                return
            if isinstance(span.span_data, HandoffSpanData):
                # Managing handoff
                if span.span_data.type == "handoff":
                    container = self.__get_container(span.trace_id, span.parent_id)
                    if container is not None:
                        from_agent = span.span_data.from_agent
                        to_agent = span.span_data.to_agent
                        container.add_event(
                            span.span_id,
                            f"Handoff {from_agent} -->  {to_agent}",
                            {"oaa_type": "handoff"},
                        )
                return
            logging.info(
                "[MaximSDK] Couldn't log span end as container is empty. Please raise an issue with Maxim engineering team if required."
            )
            return
        container.end()
        self.containers.pop(span.span_id)

    def on_trace_end(self, trace: Trace) -> None:
        container = self.__get_container(trace.trace_id)
        if container is None:
            logging.info(
                "[MaximSDK] Couldn't log trace end as container is empty. Please raise an issue with Maxim engineering team if required."
            )
            return
        container.end()
        self.containers.pop(trace.trace_id)


    def shutdown(self) -> None:
        self.logger.flush()

    def force_flush(self) -> None:
        self.logger.flush()
