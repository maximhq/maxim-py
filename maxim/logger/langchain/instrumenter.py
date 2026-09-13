from __future__ import annotations

import functools
from typing import Any, Callable, Optional
from uuid import uuid4

from ..decorators.trace import current_trace
from ..decorators.span import current_span
from ..logger import (
    GenerationConfig,
    Logger,
    RetrievalConfig,
    SpanConfig,
    TraceConfig,
    ToolCallConfig,
    ToolCallError,
)
from ..scribe import scribe
from .utils import (
    parse_base_message_to_maxim_generation,
    parse_langchain_llm_error,
    parse_langchain_llm_result,
    parse_langchain_messages,
    parse_langchain_model_parameters,
    parse_langchain_provider,
)


def _patch_method(cls: type, method_name: str, wrapper_fn: Callable[[Callable], Callable]):
    """Apply wrapper_fn to cls.method_name if present."""
    if not hasattr(cls, method_name):
        return
    orig = getattr(cls, method_name)
    if getattr(orig, "_maxim_patched", False):
        return
    wrapped = wrapper_fn(orig)
    setattr(wrapped, "_maxim_patched", True)
    setattr(cls, method_name, functools.wraps(orig)(wrapped))


def instrument_langchain(logger: Logger) -> None:
"""Instrument LangChain classes for automatic tracing.

    The helper monkey-patches core LangChain classes so that LLM calls are
    recorded as Maxim generations and chain or agent executions become spans.
    If no trace is active, a temporary trace is created around the call so logs
    are always captured.
    """

    try:  # Base language model
        from langchain_core.language_models.base import BaseLanguageModel
    except Exception:  # pragma: no cover - langchain may not be installed
        return

    try:
        from langchain.chains.base import Chain
    except Exception:  # pragma: no cover - optional
        Chain = None

    try:
        from langchain.agents.agent import AgentExecutor
    except Exception:  # pragma: no cover - optional
        AgentExecutor = None

    def _wrap_llm(method: Callable, async_fn: bool) -> Callable:
        if async_fn:
            async def async_wrapper(self, *args, **kwargs):
                messages = args[0] if args else kwargs.get("messages")
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get(
                    "metadata"
                )
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                gen_name = maxim_meta.get("generation_name") if maxim_meta else None

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                generation = None
                try:
                    model, params = parse_langchain_model_parameters(**kwargs)
                    provider = parse_langchain_provider({"name": self.__class__.__name__})
                    parsed_messages = parse_langchain_messages(messages) if messages is not None else []
                    generation = trace.generation(
                        GenerationConfig(
                            id=str(uuid4()),
                            provider=provider,
                            model=model,
                            name=gen_name,
                            model_parameters=params,
                            messages=parsed_messages,
                        )
                    )
                except Exception as e:  # pragma: no cover - best effort
                    scribe().warning(
                        f"[MaximSDK] Failed preparing langchain generation: {e}"
                    )

                try:
                    result = await method(self, *args, **kwargs)
                except Exception as err:
                    if generation is not None:
                        try:
                            generation.error(parse_langchain_llm_error(err))
                        except Exception:
                            pass
                    if is_local:
                        trace.end()
                    raise

                try:
                    if generation is not None:
                        from langchain_core.messages import BaseMessage
                        from langchain_core.outputs import LLMResult

                        if isinstance(result, LLMResult):
                            generation.result(parse_langchain_llm_result(result))
                        elif isinstance(result, BaseMessage):
                            generation.result(
                                parse_base_message_to_maxim_generation(result)
                            )
                        else:
                            generation.result(result)
                    if is_local:
                        if isinstance(result, str):
                            trace.set_output(result)
                        elif hasattr(result, "content"):
                            trace.set_output(getattr(result, "content", ""))
                        trace.end()
                except Exception as e:  # pragma: no cover - best effort
                    scribe().warning(
                        f"[MaximSDK] Failed logging langchain generation: {e}"
                    )
                return result

            return async_wrapper
        else:
            def sync_wrapper(self, *args, **kwargs):
                messages = args[0] if args else kwargs.get("messages")
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get(
                    "metadata"
                )
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                gen_name = maxim_meta.get("generation_name") if maxim_meta else None

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                generation = None
                try:
                    model, params = parse_langchain_model_parameters(**kwargs)
                    provider = parse_langchain_provider({"name": self.__class__.__name__})
                    parsed_messages = parse_langchain_messages(messages) if messages is not None else []
                    generation = trace.generation(
                        GenerationConfig(
                            id=str(uuid4()),
                            provider=provider,
                            model=model,
                            name=gen_name,
                            model_parameters=params,
                            messages=parsed_messages,
                        )
                    )
                except Exception as e:  # pragma: no cover - best effort
                    scribe().warning(
                        f"[MaximSDK] Failed preparing langchain generation: {e}"
                    )

                try:
                    result = method(self, *args, **kwargs)
                except Exception as err:
                    if generation is not None:
                        try:
                            generation.error(parse_langchain_llm_error(err))
                        except Exception:
                            pass
                    if is_local:
                        trace.end()
                    raise

                try:
                    if generation is not None:
                        from langchain_core.messages import BaseMessage
                        from langchain_core.outputs import LLMResult

                        if isinstance(result, LLMResult):
                            generation.result(parse_langchain_llm_result(result))
                        elif isinstance(result, BaseMessage):
                            generation.result(
                                parse_base_message_to_maxim_generation(result)
                            )
                        else:
                            generation.result(result)
                    if is_local:
                        if isinstance(result, str):
                            trace.set_output(result)
                        elif hasattr(result, "content"):
                            trace.set_output(getattr(result, "content", ""))
                        trace.end()
                except Exception as e:  # pragma: no cover - best effort
                    scribe().warning(
                        f"[MaximSDK] Failed logging langchain generation: {e}"
                    )
                return result

            return sync_wrapper

    def _wrap_span(method: Callable, async_fn: bool) -> Callable:
        if async_fn:
            async def async_wrapper(self, *args, **kwargs):
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get(
                    "metadata"
                )
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                span_name = maxim_meta.get("span_name") if maxim_meta else method.__qualname__

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                span = trace.span(SpanConfig(id=str(uuid4()), name=span_name))
                try:
                    result = await method(self, *args, **kwargs)
                finally:
                    span.end()
                    if is_local:
                        trace.end()
                return result

            return async_wrapper
        else:
            def sync_wrapper(self, *args, **kwargs):
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get(
                    "metadata"
                )
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                span_name = maxim_meta.get("span_name") if maxim_meta else method.__qualname__

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                span = trace.span(SpanConfig(id=str(uuid4()), name=span_name))
                try:
                    result = method(self, *args, **kwargs)
                finally:
                    span.end()
                    if is_local:
                        trace.end()
                return result

            return sync_wrapper

    def _wrap_tool(method: Callable, async_fn: bool) -> Callable:
        if async_fn:
            async def async_wrapper(self, *args, **kwargs):
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get(
                    "metadata"
                )
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                tool_name = maxim_meta.get("tool_name") if maxim_meta else getattr(self, "name", method.__qualname__)

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                parent = current_span() or trace
                tool_call = parent.tool_call(
                    ToolCallConfig(
                        id=str(uuid4()),
                        name=tool_name,
                        description=getattr(self, "description", ""),
                        args=str(args[0]) if args else "",
                    )
                )
                try:
                    result = await method(self, *args, **kwargs)
                except Exception as err:
                    tool_call.error(ToolCallError(message=str(err)))
                    if is_local:
                        trace.end()
                    raise
                else:
                    tool_call.result(result if isinstance(result, str) else str(result))
                    if is_local:
                        trace.end()
                    return result

            return async_wrapper
        else:
            def sync_wrapper(self, *args, **kwargs):
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get(
                    "metadata"
                )
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                tool_name = maxim_meta.get("tool_name") if maxim_meta else getattr(self, "name", method.__qualname__)

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                parent = current_span() or trace
                tool_call = parent.tool_call(
                    ToolCallConfig(
                        id=str(uuid4()),
                        name=tool_name,
                        description=getattr(self, "description", ""),
                        args=str(args[0]) if args else "",
                    )
                )
                try:
                    result = method(self, *args, **kwargs)
                except Exception as err:
                    tool_call.error(ToolCallError(message=str(err)))
                    if is_local:
                        trace.end()
                    raise
                else:
                    tool_call.result(result if isinstance(result, str) else str(result))
                    if is_local:
                        trace.end()
                    return result

            return sync_wrapper

    def _wrap_retriever(method: Callable, async_fn: bool) -> Callable:
        if async_fn:
            async def async_wrapper(self, *args, **kwargs):
                query = args[0] if args else kwargs.get("query")
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get("metadata")
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                name = maxim_meta.get("retrieval_name") if maxim_meta else getattr(self, "name", method.__qualname__)

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                parent = current_span() or trace
                retrieval = parent.retrieval(RetrievalConfig(id=str(uuid4()), name=name))
                if query is not None:
                    retrieval.input(query)
                try:
                    result = await method(self, *args, **kwargs)
                finally:
                    try:
                        docs = [getattr(d, "page_content", str(d)) for d in result] if result is not None else []
                        retrieval.output(docs)
                    except Exception:
                        pass
                    if is_local:
                        trace.end()
                return result

            return async_wrapper
        else:
            def sync_wrapper(self, *args, **kwargs):
                query = args[0] if args else kwargs.get("query")
                metadata = kwargs.get("config", {}).get("metadata") or kwargs.get("metadata")
                maxim_meta = metadata.get("maxim") if isinstance(metadata, dict) else None
                trace_id = maxim_meta.get("trace_id") if maxim_meta else None
                name = maxim_meta.get("retrieval_name") if maxim_meta else getattr(self, "name", method.__qualname__)

                is_local = trace_id is None and current_trace() is None
                final_trace = trace_id or (current_trace().id if current_trace() else str(uuid4()))
                trace = current_trace() or logger.trace(TraceConfig(id=final_trace))
                parent = current_span() or trace
                retrieval = parent.retrieval(RetrievalConfig(id=str(uuid4()), name=name))
                if query is not None:
                    retrieval.input(query)
                try:
                    result = method(self, *args, **kwargs)
                finally:
                    try:
                        docs = [getattr(d, "page_content", str(d)) for d in result] if result is not None else []
                        retrieval.output(docs)
                    except Exception:
                        pass
                    if is_local:
                        trace.end()
                return result

            return sync_wrapper

    _patch_method(BaseLanguageModel, "invoke", lambda m: _wrap_llm(m, False))
    _patch_method(BaseLanguageModel, "ainvoke", lambda m: _wrap_llm(m, True))
    for meth in [
        "generate",
        "agenerate",
        "predict",
        "apredict",
        "predict_messages",
        "apredict_messages",
        "stream",
        "astream",
        "stream_messages",
        "astream_messages",
        "batch",
        "abatch",
    ]:
        _patch_method(BaseLanguageModel, meth, lambda m, a=(meth.startswith("a")): _wrap_llm(m, a))

    if Chain is not None:
        _patch_method(Chain, "invoke", lambda m: _wrap_span(m, False))
        _patch_method(Chain, "ainvoke", lambda m: _wrap_span(m, True))

    if AgentExecutor is not None:
        _patch_method(AgentExecutor, "invoke", lambda m: _wrap_span(m, False))
        _patch_method(AgentExecutor, "ainvoke", lambda m: _wrap_span(m, True))

    try:
        from langchain_core.tools import BaseTool
    except Exception:  # pragma: no cover - optional
        BaseTool = None

    if BaseTool is not None:
        _patch_method(BaseTool, "run", lambda m: _wrap_tool(m, False))
        _patch_method(BaseTool, "arun", lambda m: _wrap_tool(m, True))

    try:
        from langchain_core.retrievers import BaseRetriever
    except Exception:  # pragma: no cover - optional
        BaseRetriever = None

    if BaseRetriever is not None:
        _patch_method(BaseRetriever, "get_relevant_documents", lambda m: _wrap_retriever(m, False))
        _patch_method(BaseRetriever, "aget_relevant_documents", lambda m: _wrap_retriever(m, True))
