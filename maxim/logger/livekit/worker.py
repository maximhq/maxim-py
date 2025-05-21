import functools
import inspect
import traceback

from ...scribe import scribe


def intercept_once(self, *args, **kwargs):
    action = args[0]
    if action == "worker_started":
        scribe().info(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker started")
    elif action == "worker_stopped":
        scribe().info(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker stopped")
    elif action == "worker_error":
        scribe().error(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker error")
    elif action == "worker_status_changed":
        scribe().info(
            f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker status changed"
        )


def intercept_on(self, *args, **kwargs):
    action = args[0]
    if action == "worker_started":
        scribe().info(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker started")
    elif action == "worker_stopped":
        scribe().info(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker stopped")
    elif action == "worker_error":
        scribe().error(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker error")
    elif action == "worker_status_changed":
        scribe().info(
            f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker status changed"
        )


def intercept_emit(self, *args, **kwargs):
    action = args[0]
    if action == "worker_started":
        scribe().info(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker started")
    elif action == "worker_stopped":
        scribe().info(f"[MaximSDK][LiveKit:{self.__class__.__name__}] Worker stopped")


def pre_hook(self, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
        )
        if hook_name == "emit":
            intercept_emit(self, *args, **kwargs)
        elif hook_name == "on":
            intercept_on(self, *args, **kwargs)
        elif hook_name == "once":
            intercept_once(self, *args, **kwargs)
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} completed; result={result}"
        )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_worker(orig, name):
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
