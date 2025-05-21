import functools

from ...scribe import scribe


def handle_agent(orig, name):
    def wrapper(self, *args, **kwargs):
        scribe().debug(
            f"[{self.__class__.__name__}] {name} called; args={args}, kwargs={kwargs}"
        )
        return orig(self, *args, **kwargs)

    return functools.wraps(orig)(wrapper)
