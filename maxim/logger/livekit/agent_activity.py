import functools
import inspect
import traceback

from livekit.agents.voice.agent_activity import AgentActivity

from ...scribe import scribe
from .store import get_session_store

agent_activity_f_skip_list = []


def post_start(self: AgentActivity, *args, **kwargs):
    scribe().debug(f"[{self.__class__.__name__}] post start called")
    # Trying to get AgentSession and RealtimeSession handles
    session_id = id(self._session)
    rt_session_id = id(self._rt_session)
    print(f"sessionid: {session_id}")
    print(f"rt_sessionid: {rt_session_id}")
    session_info = get_session_store().get_session_by_session_id(id(self._session))
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session start")
        return
    scribe().debug(f"[{self.__class__.__name__}] session info: {session_info}")
    session_info["rt_session_id"] = rt_session_id
    session_info["rt_session"] = self._rt_session
    get_session_store().set_session(session_info)

def push_audio(self, *args, **kwargs):
    pass

def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "push_audio":
            push_audio(self, *args, **kwargs)
            return
        elif hook_name == "_on_metrics_collected":
            pass
        else:
            scribe().debug(
                f"[{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        if hook_name == "start":
            post_start(self, *args, **kwargs)
        elif hook_name == "push_audio":
            pass
        elif hook_name == "_on_metrics_collected":
            pass
        else:
            scribe().debug(
                f"[{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_agent_activity(orig, name):
    if name in agent_activity_f_skip_list:
        return orig

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
