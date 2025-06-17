import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict, List, Optional,Union
from uuid import uuid4

from ..logger.components.span import Span, SpanConfig
from ..logger.logger import Logger
from .trace import current_logger, current_trace


@dataclass
class _SpanStack:
    """Stack of spans."""
    _stack: List[Span] = field(default_factory=list)

    def push(self, span: Span) -> None:
        """Push a span onto the stack."""
        self._stack.append(span)

    def pop(self) -> Optional[Span]:
        """Pop a span from the stack."""
        return self._stack.pop() if self._stack else None

    def current(self) -> Optional[Span]:
        """Get the current span from the stack."""
        return self._stack[-1] if self._stack else None


_span_ctx_var: ContextVar[_SpanStack] = ContextVar(
    "maxim_ctx_span_stack", default=_SpanStack([])
)


def current_span() -> Optional[Span]:
    """Get the current span from the stack."""
    return _span_ctx_var.get().current()


@contextmanager
def _push_span(span: Span):
    """Push a span onto the stack."""
    stack = _span_ctx_var.get()
    token = _span_ctx_var.set(_SpanStack([*stack._stack, span]))
    try:
        yield span
    finally:
        """Reset the span stack."""
        _span_ctx_var.reset(token)


def span(
    logger: Optional[Logger] = None,
    id: Optional[Union[str,Callable]] = None,
    trace_id: Optional[Union[str,Callable]] = None,
    name: Optional[str] = None,
    tags: Optional[dict] = None,
    evaluators: Optional[List[str]] = None,
    evaluator_variables: Optional[Dict[str, str]] = None,
):
    """
    Decorator for creating a span within a trace.

    This decorator should be used within a function that is already decorated with @trace.
    It creates a new span and injects a tracer object into the decorated function.

    Args:
        logger (Logger): The Logger instance to use for logging.
        id (Optional[str] or Optional[Callable], optional): The ID for the span. If callable, it will be called to generate the ID. Defaults to None.
        trace_id (Optional[str] or Optional[Callable], optional): The trace ID to associate with this span. If callable, it will be called to generate the trace ID. Defaults to current_trace.
        name (Optional[str], optional): The name of the span. Defaults to None.
        tags (Optional[dict], optional): Additional tags to associate with the span. Defaults to None.

    Returns:
        Callable: A decorator function that wraps the original function with span functionality.

    Raises:
        ValueError: If the decorator is used outside of a @trace decorated function.
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # First check if the logger is available
                maxim_logger = logger or current_logger()
                if maxim_logger is None:
                    raise ValueError(
                        "[MaximSDK]: no logger found. either call this function from a @trace decorated function or pass a logger"
                    )
                # Checking if its a sub-span
                parent_span = current_span()
                # We will try to get the current trace id
                actual_trace_id = trace_id() if callable(trace_id) else trace_id
                if actual_trace_id is None and (trace := current_trace()) is not None:
                    actual_trace_id = trace.id
                if actual_trace_id is None:
                    if maxim_logger.raise_exceptions:
                        raise ValueError(
                            "[MaximSDK]: no trace_id found. either call this function from a @trace decorated function or pass a trace_id"
                        )
                    else:
                        logging.warning(
                            "[MaximSDK]: no trace_id found. either call this function from a @trace decorated function or pass a trace_id"
                        )
                    return
                actual_span_id = id() if callable(id) else id
                if actual_span_id is None:
                    actual_span_id = str(uuid4())
                # Creating span
                span_config = SpanConfig(id=actual_span_id, name=name, tags=tags)
                span: Span
                if parent_span is not None:
                    span = parent_span.span(span_config)
                else:
                    span = maxim_logger.trace_add_span(
                        actual_trace_id, config=span_config
                    )
                if evaluators is not None:
                    span.evaluate().with_evaluators(*evaluators).with_variables(
                        evaluator_variables if evaluator_variables is not None else {}
                    )
                # If actual_trace_id is None, we will try to get the current trace id
                try:                    
                    with _push_span(span):
                        return await func(*args, **kwargs)
                finally:
                    span.end()
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # First check if the logger is available
                maxim_logger = logger or current_logger()
                if maxim_logger is None:
                    raise ValueError(
                        "[MaximSDK]: no logger found. either call this function from a @trace decorated function or pass a logger"
                    )
                # Checking if its a sub-span
                parent_span = current_span()
                # We will try to get the current trace id
                actual_trace_id = trace_id() if callable(trace_id) else trace_id
                if actual_trace_id is None and (trace := current_trace()) is not None:
                    actual_trace_id = trace.id
                if actual_trace_id is None:
                    if maxim_logger.raise_exceptions:
                        raise ValueError(
                            "[MaximSDK]: no trace_id found. either call this function from a @trace decorated function or pass a trace_id"
                        )
                    else:
                        logging.warning(
                            "[MaximSDK]: no trace_id found. either call this function from a @trace decorated function or pass a trace_id"
                        )
                    return
                actual_span_id = id() if callable(id) else id
                if actual_span_id is None:
                    actual_span_id = str(uuid4())
                # Creating span
                span_config = SpanConfig(id=actual_span_id, name=name, tags=tags)
                span: Span
                if parent_span is not None:
                    span = parent_span.span(span_config)
                else:
                    span = maxim_logger.trace_add_span(
                        actual_trace_id, config=span_config
                    )
                if evaluators is not None:
                    span.evaluate().with_evaluators(*evaluators).with_variables(
                        evaluator_variables if evaluator_variables is not None else {}
                    )
                # If actual_trace_id is None, we will try to get the current trace id
                try:
                    with _push_span(span):
                        return func(*args, **kwargs)
                finally:
                    span.end()
            return sync_wrapper

    return decorator
