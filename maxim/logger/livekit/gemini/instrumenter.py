"""Gemini Realtime instrumentation bootstrap.

This module activates instrumentation for Google Gemini's `RealtimeSession` by
wrapping all callable attributes (excluding dunders) with a wrapper that fires
`pre_hook` and `post_hook` around the original method. See
`gemini_realtime_session.py` for the hook implementations.

When and why this module is used
--------------------------------
- Call `instrument_gemini()` once during application startup (after the Gemini
  plugin is importable). It monkey-patches the SDK class in-place.
- Doing this at runtime avoids maintaining a fork of the upstream SDK while
  still capturing detailed observability and logging data for each session.
"""

# Import instrument_gemini_session conditionally to avoid dependency issues


def instrument_gemini():
    """Monkey-patch Gemini's `RealtimeSession` methods with instrumentation.
    """
    try:
        try:
            from livekit.plugins.google.realtime.realtime_api import RealtimeSession
        except ImportError:
            from livekit.plugins.google.beta.realtime.realtime_api import RealtimeSession
        from maxim.logger.livekit.gemini.gemini_realtime_session import instrument_gemini_session

        for name, orig in [
            (n, getattr(RealtimeSession, n))
            for n in dir(RealtimeSession)
            if callable(getattr(RealtimeSession, n))
        ]:
            if name != "__class__" and not name.startswith("__"):
                setattr(RealtimeSession, name, instrument_gemini_session(orig, name))
    except (ImportError, NameError):
        # Gemini dependencies not available, skip instrumentation
        pass
