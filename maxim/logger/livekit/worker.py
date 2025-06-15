import functools
import inspect
import traceback

from livekit.agents import Worker

from .utils import get_thread_pool_executor

from ...scribe import scribe
from .store import get_session_store


def intercept_once(self, *args, **kwargs):
    action = args[0]
    if action == "worker_started":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker started"
        )
    elif action == "worker_stopped":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker stopped"
        )
    elif action == "worker_error":
        scribe().warning(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker error"
        )
    elif action == "worker_status_changed":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker status changed"
        )


def intercept_on(self: Worker, *args, **kwargs):
    action = args[0]
    if action == "worker_started":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker started"
        )
    elif action == "worker_stopped":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker stopped"
        )
    elif action == "worker_error":
        scribe().warning(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker error"
        )
    elif action == "worker_status_changed":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker status changed"
        )


def intercept_emit(self, *args, **kwargs):
    action = args[0]
    if action == "worker_started":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker started"
        )
    elif action == "worker_stopped":
        scribe().info(
            f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker stopped"
        )


def intercept_aclose(self, *args, **kwargs):
    scribe().info(
        f"[Internal][MaximSDK][LiveKit:{self.__class__.__name__}] Worker aclose called"
    )
    # Here we will pick all ongoing sessions and end them
    get_thread_pool_executor().shutdown(wait=True)
    for session_info in get_session_store().get_all_sessions():
        scribe().debug(f"[MaximSDK] Closing session {session_info.mx_session_id}")
        get_session_store().close_session(session_info)


def pre_hook(self, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
        )
        if hook_name == "emit":
            intercept_emit(self, *args, **kwargs)
        elif hook_name == "on":
            intercept_on(self, *args, **kwargs)
        elif hook_name == "once":
            intercept_once(self, *args, **kwargs)
        elif hook_name == "aclose":
            intercept_aclose(self, *args, **kwargs)
    except Exception as e:
        scribe().warning(
            f"[MaximSDK][LiveKit:{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] {hook_name} completed; result={result}"
        )
    except Exception as e:
        scribe().warning(
            f"[MaximSDK][LiveKit:{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_worker(orig, name):
    if inspect.iscoroutinefunction(orig):

        async def async_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = await orig(self, *args, **kwargs)
                return result
            finally:
                post_hook(self, result, name, args, kwargs)

        wrapper = async_wrapper
    else:

        def sync_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = orig(self, *args, **kwargs)
                return result
            finally:
                post_hook(self, result, name, args, kwargs)

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
