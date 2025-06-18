import asyncio
import json
import logging
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..logger.components.tool_call import ToolCall, ToolCallConfig
from ..logger.logger import Logger
from .span import current_span
from .trace import current_logger, current_trace

_tool_call_ctx_var: ContextVar[Optional[ToolCall]] = ContextVar(
    "maxim_ctx_tool_call", default=None
)


def current_tool_call() -> Optional[ToolCall]:
    """Get the current tool call from the context variable."""
    return _tool_call_ctx_var.get()


def tool_call(
    logger: Optional[Logger] = None,
    id: Optional[Union[str,Callable]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    arguments: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    evaluators: Optional[List[str]] = None,
    evaluator_variables: Optional[Dict[str, str]] = None,
):
    """Decorator for tracking tool calls.

    This decorator wraps functions to automatically create and manage ToolCall
    objects for tracking tool calls, including inputs, outputs, and metadata.
    The decorated function must be called within a @trace or @span decorated context.
    """

    def decorator(func):
        """Decorator for tracking tool calls.

        Args:
            logger (Optional[Logger]): Maxim logger instance. If None, uses the current
                logger from context.
            id (Optional[str] or Optional[Callable], optional): The ID for the tool call. If callable, it will be called to generate the ID. Defaults to None.
            name (Optional[str], optional): The name of the tool call. Defaults to None.
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # First check if the logger is available
                maxim_logger = logger or current_logger()
                if maxim_logger is None:
                    raise ValueError(
                        "[MaximSDK]: no logger found. either call this function from a @trace decorated function or pass a logger"
                    )
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
                actual_tool_call_id = id() if callable(id) else id
                if actual_tool_call_id is None:
                    actual_tool_call_id = str(uuid4())
                config = ToolCallConfig(
                    id=actual_tool_call_id,
                    name=name or "",
                    description=description or "",
                    args=arguments or "",
                    tags=tags or {},
                )
                tool_call: Optional[ToolCall] = None
                if (span := current_span()) is not None:
                    tool_call = span.tool_call(config)
                elif (trace := current_trace()) is not None:
                    tool_call = trace.tool_call(config)
                if tool_call is None:
                    if maxim_logger.raise_exceptions:
                        raise ValueError(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    else:
                        logging.warning(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    return
                if evaluators is not None:
                    tool_call.evaluate().with_evaluators(*evaluators).with_variables(evaluator_variables if evaluator_variables is not None else {})
                token = _tool_call_ctx_var.set(tool_call)
                try:
                    result = await func(*args, **kwargs)
                    # Here we will check if tool_call has output
                    if isinstance(result, (dict)):
                        tool_call.result(json.dumps(result))
                    elif isinstance(result, list):
                        tool_call.result(
                            json.dumps([json.dumps(item) for item in result])
                        )
                    elif isinstance(result, str):
                        tool_call.result(result)
                    return result
                finally:
                    tool_call.end()
                    _tool_call_ctx_var.reset(token)
            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                """Synchronous wrapper for tool call tracking.

                Args:
                    args (Any): The positional arguments passed to the decorated function.
                    kwargs (Dict[str, Any]): The keyword arguments passed to the decorated function.

                Returns:
                    Any: The result of the decorated function.
                """
                # First check if the logger is available
                maxim_logger = logger or current_logger()
                if maxim_logger is None:
                    raise ValueError(
                        "[MaximSDK]: no logger found. either call this function from a @trace decorated function or pass a logger"
                    )
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
                actual_tool_call_id = id() if callable(id) else id
                if actual_tool_call_id is None:
                    actual_tool_call_id = str(uuid4())
                config = ToolCallConfig(
                    id=actual_tool_call_id,
                    name=name or "",
                    description=description or "",
                    args=arguments or "",
                    tags=tags or {},
                )
                tool_call: Optional[ToolCall] = None
                if (span:=current_span()) is not None:
                    tool_call = span.tool_call(config)
                elif (trace := current_trace()) is not None:
                    tool_call = trace.tool_call(config)
                if tool_call is None:
                    if maxim_logger.raise_exceptions:
                        raise ValueError(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    else:
                        logging.warning(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    return
                token = _tool_call_ctx_var.set(tool_call)
                try:
                    result = func(*args, **kwargs)
                    # Here we will check if tool_call has output
                    if isinstance(result, (dict)):
                        tool_call.result(json.dumps(result))
                    elif isinstance(result, list):
                        tool_call.result(
                            json.dumps([json.dumps(item) for item in result])
                        )
                    elif isinstance(result, str):
                        tool_call.result(result)
                    return result
                finally:
                    tool_call.end()
                    _tool_call_ctx_var.reset(token)
            return sync_wrapper

    return decorator
