import asyncio
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..logger.components.trace import Trace, TraceConfig
from ..logger.logger import Logger

_maxim_ctx_var_trace: ContextVar[Optional[Trace]] = ContextVar(
    "maxim_ctx_trace", default=None
)

_maxim_ctx_var_logger: ContextVar[Optional[Logger]] = ContextVar(
    "maxim_ctx_logger", default=None
)


def current_logger() -> Optional[Logger]:
    try:
        return _maxim_ctx_var_logger.get()
    except LookupError:
        return None


def current_trace() -> Optional[Trace]:
    try:
        return _maxim_ctx_var_trace.get()
    except LookupError:
        return None


def trace(
    logger: Logger,
    id: Optional[Union[str, Callable]] = None,
    sessionId: Optional[Union[str, Callable]] = None,
    name: Optional[str] = None,
    tags: Optional[dict] = None,
    evaluators: Optional[List[str]] = None,
    evaluator_variables: Optional[Dict[str, str]] = None,
):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                actual_trace_id = id() if callable(id) else id
                if actual_trace_id is None:
                    actual_trace_id = str(uuid4())
                trace_config = TraceConfig(id=actual_trace_id, name=name, tags=tags)
                # If session_id is present, we will attach that to the trace
                actual_session_id = sessionId() if callable(sessionId) else sessionId
                if actual_session_id is not None:
                    trace_config.session_id = actual_session_id
                trace = logger.trace(config=trace_config)
                if evaluators is not None:
                    trace.evaluate().with_evaluators(*evaluators).with_variables(
                        evaluator_variables or {}
                    )
                token = _maxim_ctx_var_trace.set(trace)
                _maxim_ctx_var_logger.set(logger)
                try:
                    result = await func(*args, **kwargs)
                    if trace.data().get("output", None) is None:
                        trace.set_output(str(result))
                    return result
                finally:
                    trace.end()
                    _maxim_ctx_var_trace.reset(token)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                actual_trace_id = id() if callable(id) else id
                if actual_trace_id is None:
                    actual_trace_id = str(uuid4())
                trace_config = TraceConfig(id=actual_trace_id, name=name, tags=tags)
                # If session_id is present, we will attach that to the trace
                actual_session_id = sessionId() if callable(sessionId) else sessionId
                if actual_session_id is not None:
                    trace_config.session_id = actual_session_id
                trace = logger.trace(config=trace_config)
                if evaluators is not None:
                    trace.evaluate().with_evaluators(*evaluators).with_variables(
                        evaluator_variables or {}
                    )
                token = _maxim_ctx_var_trace.set(trace)
                _maxim_ctx_var_logger.set(logger)
                try:
                    result = func(*args, **kwargs)
                    if trace.data().get("output", None) is None:
                        trace.set_output(str(result))
                    return result
                finally:
                    trace.end()
                    _maxim_ctx_var_trace.reset(token)

            return sync_wrapper

    return decorator
