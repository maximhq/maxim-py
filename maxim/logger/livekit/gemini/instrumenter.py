from .gemini_realtime_session import instrument_gemini_session


def instrument_gemini():
    try:
        from livekit.plugins.google.beta.realtime.realtime_api import RealtimeSession

        for name, orig in [
            (n, getattr(RealtimeSession, n))
            for n in dir(RealtimeSession)
            if callable(getattr(RealtimeSession, n))
        ]:
            if name != "__class__" and not name.startswith("__"):
                setattr(RealtimeSession, name, instrument_gemini_session(orig, name))
    except ImportError:
        pass
