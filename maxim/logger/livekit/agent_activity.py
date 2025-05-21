import functools
import inspect
import traceback

from ...scribe import scribe


def intercept_post_start(self, *args, **kwargs):
    scribe().debug(f"[{self.__class__.__name__}] post start called")
    # Trying to get AgentSession and RealtimeSession handles
    print(f"sessionid: {id(self._session)}")
    print(f"rt_sessionid: {id(self._rt_session)}")


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
        if hook_name == "start":
            intercept_post_start(self, *args, **kwargs)
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_agent_activity(orig, name):
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
