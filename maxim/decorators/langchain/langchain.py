import asyncio
import logging
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, List, Optional

from ...logger.langchain import MaximLangchainTracer
from ...logger.logger import Logger
from ..span import current_span
from ..trace import current_logger, current_trace

_langchain_tracer_ctx_var: ContextVar[Optional[MaximLangchainTracer]] = ContextVar(
    "maxim_ctx_langchain_tracer", default=None
)


def langchain_callback() -> Optional[MaximLangchainTracer]:
    return _langchain_tracer_ctx_var.get(None)
    


async def async_core(func, logger, name, eval_config, tags, *args, **kwargs):
    # First check if the logger is available
    maxim_logger = logger
    if maxim_logger is None:
        if current_logger() is None:
            raise ValueError(
                "[MaximSDK]: no logger found. either call this function from a @trace decorated function or pass a logger"
            )
        maxim_logger = current_logger()
        if not isinstance(maxim_logger, Logger):
            raise TypeError("[MaximSDK]: logger must be an instance of Logger")
    # Here there should be an active span or active trace
    # If none of this is the case then we raise an error
    if current_span() is None and current_trace() is None:
        if maxim_logger.raise_exceptions:
            raise ValueError(
                "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
            )
        else:
            logging.warning(
                "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
            )
    # This is a valid call
    metadata: Dict[str, Any] = {}
    if (span := current_span()) is not None:
        metadata["span_id"] = span.id
    elif (trace := current_trace()) is not None:
        metadata["trace_id"] = trace.id
    if tags is not None:
        metadata["chain_tags"] = tags
        metadata["generation_tags"] = tags
    if name is not None:
        metadata["chain_name"] = name
        metadata["generation_name"] = name
    existing_tracer = _langchain_tracer_ctx_var.get(None)
    if not existing_tracer:
        tracer = MaximLangchainTracer(logger=maxim_logger, metadata=metadata, eval_config=eval_config)
        _langchain_tracer_ctx_var.set(tracer)
    return await func(*args, **kwargs)

def sync_core(func, logger, name, eval_config, tags, *args, **kwargs):
    # First check if the logger is available
    maxim_logger = logger
    if maxim_logger is None:
        if current_logger() is None:
            raise ValueError(
                "[MaximSDK]: no logger found. either call this function from a @trace decorated function or pass a logger"
            )
        maxim_logger = current_logger()
        if not isinstance(maxim_logger, Logger):
            raise TypeError("[MaximSDK]: logger must be an instance of Logger")
    # Here there should be an active span or active trace
    # If none of this is the case then we raise an error
    if current_span() is None and current_trace() is None:
        if maxim_logger.raise_exceptions:
            raise ValueError(
                "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
            )
        else:
            logging.warning(
                "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
            )
    # This is a valid call
    metadata: Dict[str, Any] = {}
    if (span := current_span()) is not None:
        metadata["span_id"] = span.id
    elif (trace := current_trace()) is not None:
        metadata["trace_id"] = trace.id
    if tags is not None:
        metadata["chain_tags"] = tags
        metadata["generation_tags"] = tags
    if name is not None:
        metadata["chain_name"] = name
        metadata["generation_name"] = name
    existing_tracer = _langchain_tracer_ctx_var.get(None)
    if not existing_tracer:
        tracer = MaximLangchainTracer(logger=maxim_logger, metadata=metadata, eval_config=eval_config)
        _langchain_tracer_ctx_var.set(tracer)
    return func(*args, **kwargs)


def langchain_llm_call(
    logger: Optional[Logger] = None,
    name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    eval_config: Optional[Dict[str, List[str]]] = None,
):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await async_core(
                    func, logger, name, eval_config, tags, *args, **kwargs
                )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return sync_core(func, logger, name, eval_config, tags, *args, **kwargs)
            return sync_wrapper
    return decorator


def langgraph_agent(
    logger: Optional[Logger] = None,
    name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    eval_config: Optional[Dict[str, List[str]]] = None
):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await async_core(func, logger, name, eval_config, tags, *args, **kwargs)

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return sync_core(func, logger, name, eval_config, tags, *args, **kwargs)

            return sync_wrapper

    return decorator
