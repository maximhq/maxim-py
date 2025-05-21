import functools
import inspect
import traceback

from livekit.agents import Agent, AgentSession

from ...scribe import scribe


def intercept_session_start(self: AgentSession, *args, **kwargs):
    """
    This function is called when a session starts.
    This is the point where we create a new session for Maxim.
    The session info along with room_id, agent_id, etc is stored in the thread-local store.
    """
    scribe().debug(
        f"[{self.__class__.__name__}] Session started; args={args}, kwargs={kwargs}"
    )
    # getting the room_id
    room_id = kwargs.get("room", None)
    agent: Agent = kwargs.get("agent", None)
    print(f"session key:{id(self)}")
    print(f"Room: {room_id}")
    print(f"Agent: {agent.instructions}")
    print("##############CREATING SESSION################")
    # creating trace as well
    print("##############CREATING TRACE################")
    # get_session_store().set_session(
    #     room_id,
    #     SessionStoreEntry(
    #         room_id=room_id,
    #         agent_id=agent.get("agent_id", None),
    #         session_id=session_id,
    #         current_trace_id=current_trace_id,
    #     ),
    # )


def intercept_update_agent_state(self, *args, **kwargs):
    """
    This function is called when the agent state is updated.
    """
    new_state = args[0]
    scribe().debug(
        f"[{self.__class__.__name__}] Agent state updated; new_state={new_state}"
    )
    print("###########SENDING AGENT STATE UPDATED EVENT###########")


def intercept_generate_reply(self, *args, **kwargs):
    """
    This function is called when the agent generates a reply.
    """
    instructions = kwargs.get("instructions", None)
    scribe().debug(
        f"[{self.__class__.__name__}] Reply generated; instructions={instructions} kwargs={kwargs}"
    )
    print("###########CREATING LLM CALL###########")


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "start":
            intercept_session_start(self, *args, **kwargs)
        elif hook_name == "_update_agent_state":
            intercept_update_agent_state(self, *args, **kwargs)
        elif hook_name == "generate_reply":
            intercept_generate_reply(self, *args, **kwargs)
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
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} completed; result={result}"
        )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_agent_session(orig, name):
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
