import functools

from ...scribe import scribe


def on_session_start(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session started; args={args}, kwargs={kwargs}"
    )


def on_session_end(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session ended; args={args}, kwargs={kwargs}"
    )


def on_session_error(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session error; args={args}, kwargs={kwargs}"
    )


def on_session_message(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session message; args={args}, kwargs={kwargs}"
    )


def on_session_audio(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session audio; args={args}, kwargs={kwargs}"
    )


def on_session_video(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session video; args={args}, kwargs={kwargs}"
    )


def on_session_data(self, *args, **kwargs):
    scribe().debug(
        f"[{self.__class__.__name__}] Session data; args={args}, kwargs={kwargs}"
    )


def instrument_agent_session(orig, name):
    def wrapper(self, *args, **kwargs):
        if name == "start":
            on_session_start(self, *args, **kwargs)
        elif name == "end":
            on_session_end(self, *args, **kwargs)
        elif name == "error":
            on_session_error(self, *args, **kwargs)
        else:
            scribe().debug(
                f"[{self.__class__.__name__}] {name} called; args={args}, kwargs={kwargs}"
            )
        return orig(self, *args, **kwargs)

    return functools.wraps(orig)(wrapper)
