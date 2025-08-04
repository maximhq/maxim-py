import functools
import inspect
import traceback
from uuid import uuid4

from livekit.agents import JobContext

from ...scribe import scribe
from .store import get_session_store


def intercept_participant_available(self: JobContext, participant):
    if not participant:
        return
    trace = get_session_store().get_current_trace_for_room_id(str(id(self.room)))
    if trace is None:
        return
    trace.event(
        str(uuid4()),
        "Participant available",
        {"type": "participant_available"},
        {"participant": participant},
    )


def pre_hook(self, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
        )
    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] {hook_name} completed; result={result}"
        )
        if hook_name == "_participant_available":
            if not args or len(args) == 0:
                return
            intercept_participant_available(self, args[0])

    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def instrument_job_context(orig, name):
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
