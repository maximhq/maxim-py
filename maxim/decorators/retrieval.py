import asyncio
import json
import logging
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..logger.components.retrieval import Retrieval, RetrievalConfig
from ..logger.logger import Logger
from .span import current_span
from .trace import current_logger, current_trace

_retrieval_ctx_var: ContextVar[Optional[Retrieval]] = ContextVar(
    "maxim_ctx_retrieval", default=None
)


def current_retrieval() -> Optional[Retrieval]:
    """Get the current retrieval from the context variable.

    Returns:
        Optional[Retrieval]: The current retrieval instance if one exists,
            otherwise None.
    """
    return _retrieval_ctx_var.get()


def retrieval(
    logger: Optional[Logger] = None,
    id: Optional[Union[str,Callable]] = None,
    input: Optional[Union[str, Callable]] = None,
    name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    evaluators: Optional[List[str]] = None,
    evaluator_variables: Optional[Dict[str, str]] = None,
):
    """Decorator for tracking retrieval operations.

    This decorator wraps functions to automatically create and manage Retrieval
    objects for tracking retrieval operations, including inputs, outputs, and metadata.
    The decorated function must be called within a @trace or @span decorated context.

    Args:
        logger (Optional[Logger]): Maxim logger instance. If None, uses the current
            logger from context.
        id (Optional[str] or Optional[Callable], optional): The ID for the retrieval. If callable, it will be called to generate the ID. Defaults to None.
        input (Optional[str] or Optional[Callable], optional): The input for the retrieval. If callable, it will be called to generate the input. Defaults to None.
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
                actual_retrieval_id = id() if callable(id) else id
                if actual_retrieval_id is None:
                    actual_retrieval_id = str(uuid4())
                config = RetrievalConfig(id=actual_retrieval_id, name=name, tags=tags)
                retrieval: Optional[Retrieval] = None
                if (span := current_span()) is not None:
                    retrieval = span.retrieval(config)
                elif (trace := current_trace()) is not None:
                    retrieval = trace.retrieval(config)
                if retrieval is None:
                    if maxim_logger.raise_exceptions:
                        raise ValueError(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    else:
                        logging.warning(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    return
                # Checking for input
                if evaluators is not None:
                    retrieval.evaluate().with_evaluators(*evaluators).with_variables(evaluator_variables if evaluator_variables is not None else {})
                actual_input = input() if callable(input) else input
                if actual_input is not None:
                    retrieval.input(actual_input)
                token = _retrieval_ctx_var.set(retrieval)
                try:
                    result = await func(*args, **kwargs)
                    # Here we will check if retrieval has output
                    if not retrieval.is_output_set:
                        if isinstance(result, (dict)):
                            retrieval.output(json.dumps(result))
                        elif isinstance(result, list):
                            retrieval.output([json.dumps(item) for item in result])
                        elif isinstance(result, str):
                            retrieval.output(result)
                    return result
                finally:
                    retrieval.end()
                    _retrieval_ctx_var.reset(token)
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
                actual_retrieval_id = id() if callable(id) else id
                if actual_retrieval_id is None:
                    actual_retrieval_id = str(uuid4())
                config = RetrievalConfig(id=actual_retrieval_id, name=name, tags=tags)
                retrieval: Optional[Retrieval] = None
                if (span:=current_span()) is not None:
                    retrieval = span.retrieval(config)
                elif (trace := current_trace()) is not None:
                    retrieval = trace.retrieval(config)
                if retrieval is None:
                    if maxim_logger.raise_exceptions:
                        raise ValueError(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    else:
                        logging.warning(
                            "[MaximSDK]: no trace or span found. either call this function from a @trace or @span decorated function"
                        )
                    return
                # Checking for input
                actual_input = input() if callable(input) else input
                if actual_input is not None:
                    retrieval.input(actual_input)
                token = _retrieval_ctx_var.set(retrieval)
                try:
                    result = func(*args, **kwargs)
                    # Here we will check if retrieval has output
                    if not retrieval.is_output_set:
                        if isinstance(result, (dict)):
                            retrieval.output(json.dumps(result))
                        elif isinstance(result, list):
                            retrieval.output([json.dumps(item) for item in result])
                        elif isinstance(result, str):
                            retrieval.output(result)
                    return result
                finally:
                    retrieval.end()
                    _retrieval_ctx_var.reset(token)
            return sync_wrapper

    return decorator
