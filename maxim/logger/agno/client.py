"""Agno integration for the Maxim SDK.

This module instruments the :mod:`agno` library so ``Agent.run`` and
``Agent.arun`` calls are automatically traced via :class:`~maxim.logger.Logger`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from uuid import uuid4
import inspect
from agno.run.response import RunEvent

from ..logger import GenerationConfig, Logger, TraceConfig
from ...scribe import scribe


def _start_trace(
    logger: Logger,
    agent: Any,
    trace_id: Optional[str],
    generation_name: Optional[str],
) -> tuple[Any, Any, str, bool]:
    """Create a trace and generation for the run.

    Returns the created ``trace`` and ``generation`` instances along with the
    final ``trace_id`` and a flag indicating if the trace should be ended
    automatically.
    """

    is_local_trace = trace_id is None
    final_trace_id = trace_id or str(uuid4())
    trace = logger.trace(TraceConfig(id=final_trace_id))
    gen_config = GenerationConfig(
        id=str(uuid4()),
        model=getattr(agent, "model", None),
        provider="agno",
        name=generation_name,
        messages=[],
        model_parameters={},
    )
    generation = trace.generation(gen_config)
    return trace, generation, final_trace_id, is_local_trace


def _log_event(trace: Any, generation: Any, event: Any, content: Any) -> None:
    """Log an Agno event to the trace and record results if completed."""

    if not event:
        return

    trace.add_event(str(uuid4()), str(event))

    if event in (RunEvent.run_completed, RunEvent.run_completed.value, "RunCompleted"):
        generation.result(content)


def _wrap_sync(logger: Logger, fn: Callable) -> Callable:
    """Wrap a synchronous ``Agent.run`` implementation."""

    def wrapper(
        self,
        *args: Any,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        trace, generation, _, end_trace = _start_trace(logger, self, trace_id, generation_name)
        result = None
        try:
            result = fn(self, *args, **kwargs)
            if inspect.isgenerator(result):
                def _iterate() -> Any:
                    for chunk in result:
                        _log_event(
                            trace,
                            generation,
                            getattr(chunk, "event", None),
                            getattr(chunk, "content", str(chunk)),
                        )
                        yield chunk
                    if end_trace:
                        trace.end()

                return _iterate()

            _log_event(
                trace,
                generation,
                getattr(result, "event", None),
                getattr(result, "content", str(result)),
            )
            return result
        except Exception as exc:  # pragma: no cover - passthrough
            generation.error(str(exc))
            raise
        finally:
            if end_trace and not (result is not None and inspect.isgenerator(result)):
                trace.end()

    return wrapper


def _wrap_async(logger: Logger, fn: Callable) -> Callable:
    """Wrap an asynchronous ``Agent.arun`` implementation."""

    async def wrapper(
        self,
        *args: Any,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        trace, generation, _, end_trace = _start_trace(logger, self, trace_id, generation_name)
        result = None
        try:
            result = await fn(self, *args, **kwargs)
            if inspect.isasyncgen(result):
                async def _iterate() -> Any:
                    async for chunk in result:
                        _log_event(
                            trace,
                            generation,
                            getattr(chunk, "event", None),
                            getattr(chunk, "content", str(chunk)),
                        )
                        yield chunk
                    if end_trace:
                        trace.end()

                return _iterate()

            _log_event(
                trace,
                generation,
                getattr(result, "event", None),
                getattr(result, "content", str(result)),
            )
            return result
        except Exception as exc:  # pragma: no cover - passthrough
            generation.error(str(exc))
            raise
        finally:
            if end_trace and not (result is not None and inspect.isasyncgen(result)):
                trace.end()

    return wrapper


def instrument_agno(logger: Logger) -> None:
    """Patch Agno's Agent.run and Agent.arun methods to log via Maxim."""
    try:
        from agno.agent.agent import Agent
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "Agno library is required for instrumentation"
        ) from exc

    if getattr(Agent, "_maxim_patched", False):
        return

    original_run = Agent.run
    original_arun = getattr(Agent, "arun", None)

    Agent.run = _wrap_sync(logger, original_run)
    if original_arun is not None:
        Agent.arun = _wrap_async(logger, original_arun)
    Agent._maxim_patched = True
    scribe().info("[MaximSDK] Agno instrumentation enabled")


class MaximAgnoClient:
    """Initialize Agno instrumentation for a Maxim logger."""

    def __init__(self, logger: Logger) -> None:
        """Create a client and immediately instrument ``agno``.

        Args:
            logger: The :class:`~maxim.logger.Logger` instance to record events
                with.
        """
        self._logger = logger
        try:
            instrument_agno(logger)
        except Exception as e:  # pragma: no cover - runtime errors
            scribe().warning(f"[MaximSDK][Agno] Failed to instrument Agno: {e}")


