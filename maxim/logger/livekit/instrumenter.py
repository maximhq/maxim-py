from livekit.agents import Agent, AgentSession, JobContext, Worker
from livekit.agents.llm import RealtimeModel, RealtimeSession
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.rtc import Room

from ...logger import Logger
from .agent_activity import instrument_agent_activity
from .agent_session import instrument_agent_session
from .gemini.instrumenter import instrument_gemini
from .job_context import instrument_job_context
from .realtime_session import instrument_realtime_session
from .store import MaximLiveKitCallback, set_livekit_callback, set_maxim_logger
from .worker import instrument_worker


def instrument_livekit(logger: Logger, callback: MaximLiveKitCallback = None):
    """Instrument LiveKit classes with logging.

    This function adds logging instrumentation to LiveKit classes (Agent, JobContext, LLM)
    by wrapping their methods with logging decorators. It logs method calls with their
    arguments and keyword arguments.

    The instrumentation:
    1. Wraps all Agent methods starting with "on_"
    2. Wraps all JobContext methods (except special methods)
    3. Wraps all LLM methods (except special methods)
    """
    print(
        "[MaximSDK] Warning: LiveKit instrumentation is in alpha state. Please report any issues here: https://github.com/maximhq/maxim-py/issues"
    )
    set_maxim_logger(logger)
    set_livekit_callback(callback)
    # Instrument AgentSession methods
    for name, orig in [
        (n, getattr(AgentSession, n))
        for n in dir(AgentSession)
        if callable(getattr(AgentSession, n))
    ]:
        if name != "__class__" and not name.startswith("__"):
            setattr(AgentSession, name, instrument_agent_session(orig, name))

    # Instrument Worker methods
    for name, orig in [
        (n, getattr(Worker, n)) for n in dir(Worker) if callable(getattr(Worker, n))
    ]:
        if name != "__class__" and not name.startswith("__"):
            setattr(Worker, name, instrument_worker(orig, name))

    # Instrument RealtimeSession methods
    for name, orig in [
        (n, getattr(RealtimeSession, n))
        for n in dir(RealtimeSession)
        if callable(getattr(RealtimeSession, n))
    ]:
        if name != "__class__" and not name.startswith("__"):
            setattr(RealtimeSession, name, instrument_realtime_session(orig, name))

    # Instrument AgentActivity methods
    for name, orig in [
        (n, getattr(AgentActivity, n))
        for n in dir(AgentActivity)
        if callable(getattr(AgentActivity, n))
    ]:
        if name != "__class__" and not name.startswith("__"):
            setattr(AgentActivity, name, instrument_agent_activity(orig, name))

    # Instrument JobContext methods
    for name, orig in [
        (n, getattr(JobContext, n))
        for n in dir(JobContext)
        if callable(getattr(JobContext, n))
    ]:
        if name != "__class__" and not name.startswith("__"):
            setattr(JobContext, name, instrument_job_context(orig, name))

    # Instrument gemini models if present
    instrument_gemini()