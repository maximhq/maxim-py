from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from llama_index.core.callbacks import BaseCallbackHandler, CBEventType, EventPayload
from llama_index.core.llms import ChatMessage

from ..logger import GenerationConfig, GenerationError, Logger, TraceConfig
from ..models import TraceContainer
from ...scribe import scribe


class MaximLlamaIndexCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs LlamaIndex LLM calls to Maxim."""

    def __init__(self, logger: Logger):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.logger = logger
        self._containers: Dict[str, TraceContainer] = {}

    # BaseCallbackHandler requires these methods
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(self, trace_id: Optional[str] = None, trace_map: Optional[Dict[str, List[str]]] = None) -> None:
        pass

    def _parse_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        parsed: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = str(msg.content) if msg.content is not None else ""
            parsed.append({"role": role, "content": content})
        return parsed

    def _parse_provider(self, class_name: Optional[str]) -> str:
        if not class_name:
            return "llamaindex"
        lowered = class_name.lower()
        if "openai" in lowered:
            return "openai"
        if "anthropic" in lowered:
            return "anthropic"
        if "bedrock" in lowered:
            return "bedrock"
        if "gemini" in lowered or "google" in lowered:
            return "google"
        return "llamaindex"

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type != CBEventType.LLM or payload is None:
            return event_id
        try:
            messages: Sequence[ChatMessage] = payload.get(EventPayload.MESSAGES, [])  # type: ignore
            model_info = payload.get(EventPayload.SERIALIZED, {})
            provider = self._parse_provider(model_info.get("class_name"))
            model = model_info.get("model") or model_info.get("model_name")
            params = payload.get(EventPayload.ADDITIONAL_KWARGS, {})
            trace = TraceContainer(
                logger=self.logger,
                trace_id=str(uuid4()),
                trace_name="LlamaIndex",
            )
            trace.create()
            gen_config = GenerationConfig(
                id=event_id or str(uuid4()),
                provider=provider,
                model=model or "unknown",
                model_parameters=params,
                messages=self._parse_messages(messages),
            )
            trace.add_generation(gen_config)
            self._containers[event_id] = trace
        except Exception as e:
            scribe().warning(f"[MaximSDK][LlamaIndex] Failed to process start: {e}")
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type != CBEventType.LLM:
            return
        container = self._containers.pop(event_id, None)
        if container is None:
            return
        try:
            if payload and EventPayload.RESPONSE in payload:
                self.logger.generation_result(event_id, payload[EventPayload.RESPONSE])
            elif payload and EventPayload.COMPLETION in payload:
                self.logger.generation_result(event_id, payload[EventPayload.COMPLETION])
            elif payload and EventPayload.EXCEPTION in payload:
                err = payload[EventPayload.EXCEPTION]
                self.logger.generation_error(
                    generation_id=event_id,
                    error=GenerationError(message=str(err), code="exception"),
                )
        except Exception as e:
            scribe().warning(f"[MaximSDK][LlamaIndex] Failed to log result: {e}")
        finally:
            container.end()

__all__ = ["MaximLlamaIndexCallbackHandler"]
