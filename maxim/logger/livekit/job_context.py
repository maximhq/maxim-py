import functools

from ...scribe import scribe


def instrument_job_context(orig, name):
    def wrapper(self, *args, **kwargs):
        scribe().debug(
            f"[{self.__class__.__name__}] {name} called; args={args}, kwargs={kwargs}"
        )
        return orig(self, *args, **kwargs)

    return functools.wraps(orig)(wrapper)
