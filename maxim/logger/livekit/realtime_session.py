import functools

from ...scribe import scribe


def handle_realtime_session_emit(self, *args, **kwargs):
    if args[0] == "openai_client_event_queued":
        return
    scribe().debug(
        f"[{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}"
    )


def instrument_realtime_session(orig, name):
    def wrapper(self, *args, **kwargs):
        if name == "emit":
            handle_realtime_session_emit(self, *args, **kwargs)
        else:
            scribe().debug(
                f"[{self.__class__.__name__}] {name} called; args={args}, kwargs={kwargs}"
            )
        return orig(self, *args, **kwargs)

    return functools.wraps(orig)(wrapper)
