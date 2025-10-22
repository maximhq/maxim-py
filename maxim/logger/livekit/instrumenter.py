from livekit.agents import AgentSession
from livekit.agents.llm import RealtimeSession
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.plugins.openai import LLM, STT, TTS
from ...logger import Logger
from .agent_activity import instrument_agent_activity
from .agent_session import instrument_agent_session
# Import instrument_gemini conditionally to avoid dependency issues
from .llm import instrument_llm_init
from .realtime_session import instrument_realtime_session
from .stt import instrument_stt_init
from .store import MaximLiveKitCallback, set_livekit_callback, set_maxim_logger
from .tts import instrument_tts_init


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
        "[MaximSDK] Warning: LiveKit instrumentation is in beta phase. Please report any issues here: https://github.com/maximhq/maxim-py/issues"
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
    # for name, orig in [
    #     (n, getattr(Worker, n)) for n in dir(Worker) if callable(getattr(Worker, n))
    # ]:
        # if name != "__class__" and not name.startswith("__"):
        #     setattr(Worker, name, instrument_worker(orig, name))

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
    # for name, orig in [
    #     (n, getattr(JobContext, n))
    #     for n in dir(JobContext)
    #     if callable(getattr(JobContext, n))
    # ]:
    #     if name != "__class__" and not name.startswith("__"):
    #         setattr(JobContext, name, instrument_job_context(orig, name))

    # Instrumenting LLM models if present
    for name, orig in [
        (n, getattr(LLM, n)) for n in dir(LLM) if callable(getattr(LLM, n))
    ]:
        if name == "__init__":
            setattr(
                LLM,
                name,
                instrument_llm_init(orig, name, LLM.__name__),
            )

    # Instrumenting STT models if present
    for name, orig in [
        (n, getattr(STT, n)) for n in dir(STT) if callable(getattr(STT, n))
    ]:
        if name == "__init__":
            setattr(
                STT,
                name,
                instrument_stt_init(orig, name, STT.__name__),
            )
        elif not name.startswith("__"):
            from .stt import instrument_stt

            setattr(STT, name, instrument_stt(orig, name))

    # Instrumenting TTS models if present
    for name, orig in [
        (n, getattr(TTS, n)) for n in dir(TTS) if callable(getattr(TTS, n))
    ]:
        if name == "__init__":
            setattr(
                TTS,
                name,
                instrument_tts_init(orig, name, TTS.__name__),
            )
        elif not name.startswith("__"):
            from .tts import instrument_tts

            setattr(TTS, name, instrument_tts(orig, name))

    # Instrument gemini models if present
    try:
        from .gemini.instrumenter import instrument_gemini
        instrument_gemini()
    except (ImportError, NameError):
        # Gemini dependencies not available, skip instrumentation
        pass
