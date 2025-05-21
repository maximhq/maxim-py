import functools
import inspect
import traceback

from ...scribe import scribe


def intercept_participant_available(self, *args, **kwargs):
    participant = args[0]
    scribe().debug(
        f"[{self.__class__.__name__}] new participant available; participant={participant}"
    )
    print("###########SENDING PARTICIPANT AVAILABLE EVENT###########")


def pre_hook(self, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
        )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} completed; result={result}"
        )
        if hook_name == "_participant_available":
            intercept_participant_available(self, *args, **kwargs)
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_job_context(orig, name):
    if inspect.iscoroutinefunction(orig):

        async def async_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = await orig(self, *args, **kwargs)
            post_hook(self, result, name, args, kwargs)
            return result

        wrapper = async_wrapper
    else:

        def sync_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = orig(self, *args, **kwargs)
            post_hook(self, result, name, args, kwargs)
            return result

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
