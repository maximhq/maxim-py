import functools
import inspect
import traceback
import time
from typing import Optional
import uuid
import weakref
from datetime import datetime, timezone
from io import BytesIO

from livekit.agents import Agent, AgentSession
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.plugins.openai.llm import _LLMOptions
from livekit.protocol.models import Room

from maxim.logger.components.attachment import FileDataAttachment
from maxim.logger.components.generation import (
    GenerationConfigDict,
    GenerationResult,
    GenerationResultChoice,
)
from maxim.logger.utils import pcm16_to_wav_bytes

from maxim.scribe import scribe
from maxim.logger.livekit.store import (
    SessionState,
    SessionStoreEntry,
    Turn,
    get_livekit_callback,
    get_maxim_logger,
    get_session_store,
    get_tts_store,
)
from maxim.logger.livekit.utils import extract_llm_model_parameters, extract_llm_usage, get_thread_pool_executor, start_new_turn


def intercept_session_start(self: AgentSession, room, room_name, agent: Agent):
    """
    This function is called when a session starts.
    This is the point where we create a new session for Maxim.
    The session info along with room_id, agent_id, etc is stored in the thread-local store.
    """
    maxim_logger = get_maxim_logger()

    # Wait for start signal (max ~5s) before proceeding
    for _ in range(500):
        if getattr(self, "_started", False):
            scribe().debug(f"[Internal][{self.__class__.__name__}] Session started")
            break
        time.sleep(0.01)
    else:
        scribe().debug(f"[Internal][{self.__class__.__name__}] start not signaled within timeout; continuing")
    # getting the room_id
    if isinstance(room, str):
        room_id = room
        room_name = room
    elif isinstance(room, Room):
        room_id = room.sid
        room_name = room.name
    else:
        room_id = id(room)
        if isinstance(room, dict):
            room_name = room.get("name")
    scribe().debug(f"[Internal] Session key:{id(self)}")
    scribe().debug(f"[Internal] Room: {room_id}")
    scribe().debug(f"[Internal] Agent: {agent.instructions}")
    # creating trace as well
    session_id = str(uuid.uuid4())
    session = maxim_logger.session({"id": session_id, "name": "livekit-session"})
    # adding tags to the session
    if room_id is not None:
        session.add_tag("room_id", str(room_id))
    if room_name is not None:
        session.add_tag("room_name", str(room_name))
    if session_id is not None:
        session.add_tag("session_id", str(session_id))
    if agent is not None:
        session.add_tag("agent_id", str(id(agent)))
    # If callback is set, emit the session started event
    callback = get_livekit_callback()
    if callback is not None:
        try:
            callback(
                "maxim.session.started", {"session_id": session_id, "session": session}
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] An error was captured during LiveKit callback execution: {e!s}"
            )
    trace_id = str(uuid.uuid4())
    tags: dict[str, str] = {}
    if room_id is not None:
        tags["room_id"] = str(room_id)
    if room_name is not None:
        tags["room_name"] = room_name
    tags["session_id"] = str(id(self))
    if agent is not None:
        tags["agent_id"] = str(id(agent))

    trace = session.trace(
        {
            "id": trace_id,
            "input": agent.instructions,
            "name": "Greeting turn",
            "session_id": session_id,
            "tags": tags,
        }
    )

    current_turn_id = str(uuid.uuid4())
    if self.stt is not None or agent.stt is not NOT_GIVEN:
        # Only add generation if we are not in realtime session
        llm_opts: _LLMOptions = self.llm._opts if self.llm is not None else agent.llm._opts
        model = self.llm.model if self.llm is not None else agent.llm.model
        if llm_opts is not None:
            model_parameters = extract_llm_model_parameters(llm_opts)
        else:
            model_parameters = None
        trace.generation(GenerationConfigDict(
            id=current_turn_id,
            model=model if model is not None else "unknown",
            model_parameters=model_parameters if model_parameters is not None else {},
            messages=[{"role": "system", "content": agent.instructions}],
            provider="livekit",
            name="Greeting turn",
        ))

    callback = get_livekit_callback()
    if callback is not None:
        try:
            callback("maxim.trace.started", {"trace_id": trace_id, "trace": trace})
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] An error was captured during LiveKit callback execution: {e!s}"
            )

    current_turn = Turn(
        turn_id=current_turn_id,
        turn_sequence=0,
        turn_timestamp=datetime.now(timezone.utc),
        is_interrupted=False,
        turn_input_transcription="",
        turn_output_transcription="",
        turn_input_audio_buffer=BytesIO(),
        turn_output_audio_buffer=BytesIO(),
    )
    session_to_set = SessionStoreEntry(
            room_id=room_id,
            user_speaking=False,
            provider="unknown",
            conversation_buffer=BytesIO(),
            conversation_buffer_index=1,
            state=SessionState.INITIALIZED,
            agent_id=id(agent),
            room_name=room_name,
            agent_session_id=id(self),
            agent_session=weakref.ref(self),
            rt_session_id=None,
            rt_session=None,
            llm_config=None,
            rt_session_info={},
            mx_current_trace_id=trace_id,
            mx_session_id=session_id,
            current_turn=current_turn,
        )

    scribe().debug(f"[Internal] Session to set: {session_to_set}")

    get_session_store().set_session(
        session_to_set
    )

def intercept_update_agent_state(self: AgentSession, new_state):
    """
    This function is called when the agent state is updated.
    """
    if new_state is None:
        return
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.event(
            str(uuid.uuid4()),
            f"agent_{new_state}",
            {"new_state": new_state, "platform": "livekit"},
        )


def intercept_generate_reply(self: AgentSession, instructions):
    """
    This function is called when the agent generates a reply.
    """
    if instructions is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] Generate reply; instructions={instructions}"
    )
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.set_input(instructions)


def intercept_user_state_changed(self: AgentSession, new_state):
    """
    This function is called when the user state is changed.
    """
    if new_state is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] User state changed; new_state={new_state}"
    )
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.event(
            str(uuid.uuid4()),
            f"user_{new_state}",
            {"new_state": new_state, "platform": "livekit"},
        )


def handle_tool_call_executed(self: AgentSession, event: FunctionToolsExecutedEvent):
    """
    This function is called when the agent executes a tool call.
    """
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is None:
        return
    # this we consider as a tool call result event
    # tool call creation needs to be done at each provider level
    for function_call in event.function_calls:
        tool_call = trace.tool_call(
            {
                "id": function_call.call_id,
                "name": function_call.name,
                "description": "",
                "args": (
                    str(function_call.arguments)
                    if function_call.arguments is not None
                    else ""
                ),
            }
        )
        tool_output = ""
        for output in event.function_call_outputs or []:
            if output is not None and output.call_id == function_call.call_id:
                tool_output = output.output
                break
        tool_call.result(tool_output)


def handle_agent_response_complete(self: AgentSession, response_text):
    """Handle agent response completion and attach output audio"""
    try:
        session_info = get_session_store().get_session_by_agent_session_id(
            id(self)
        )
        if session_info is None:
            return

        if session_info.rt_session_id is not None:
            return

        turn = session_info.current_turn
        if turn is None:
            return

        llm_opts: _LLMOptions = self.llm._opts if self.llm is not None else self._agent.llm._opts
        if llm_opts is not None:
            model_parameters = extract_llm_model_parameters(llm_opts)
        else:
            model_parameters = None

        model = self.llm.model if self.llm is not None else self._agent.llm.model

        usage = extract_llm_usage(id(self), self.llm if self.llm is not None else self._agent.llm)

        tts_id = id(self.tts) if self.tts is not None else None
        tts_audio_frames = get_tts_store().get_tts_audio_data(tts_id) if tts_id is not None else None

        if tts_audio_frames is not None and len(tts_audio_frames) > 0:
            for frame in tts_audio_frames:
                turn.turn_output_audio_buffer.write(frame.data)
                session_info.conversation_buffer.write(frame.data)

        get_tts_store().clear_tts_audio_data(tts_id)

        if response_text:
            turn.turn_output_transcription = response_text

            # Add output audio attachment if we have a generation and audio
            if (
                turn.turn_output_audio_buffer is not None
                and turn.turn_output_audio_buffer.tell() > 0
            ):
                get_maxim_logger().generation_add_attachment(
                    turn.turn_id,
                    FileDataAttachment(
                        data=pcm16_to_wav_bytes(
                            turn.turn_output_audio_buffer.getvalue()
                        ),
                        tags={"attach-to": "output"},
                        name="Agent Audio Response",
                        timestamp=int(time.time()),
                    ),
                )

            # Update trace output
            trace = get_session_store().get_current_trace_for_agent_session(
                id(self)
            )
            if trace is not None:
                trace.set_output(response_text)

            choices: list[GenerationResultChoice] = []
            choice: GenerationResultChoice = {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "audio", "transcript": response_text}],
                    "tool_calls": [],
                },
            }
            choices.append(choice)
            result = GenerationResult(
                id=str(uuid.uuid4()),
                object="tts.response",
                created=int(time.time()),
                model=model if model is not None else "unknown",
                choices=choices,
                usage=usage if usage is not None else { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
            )
            try:
                get_maxim_logger().generation_result(turn.turn_id, result)
                get_maxim_logger().generation_set_model(
                    turn.turn_id,
                    model if model is not None else "unknown"
                )
                get_maxim_logger().generation_set_model_parameters(
                    turn.turn_id,
                    model_parameters if model_parameters is not None else {}
                )
            except Exception as e:
                scribe().warning(
                    f"[MAXIM SDK] Error adding generation result; error={e!s}\n{traceback.format_exc()}"
                )

            session_info.current_turn = turn
            get_session_store().set_session(session_info)

    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] agent response handling failed; error={e!s}\n{traceback.format_exc()}"
        )


def intercept_metrics_collected(self, event):
    """
    This function is called when the metrics are collected.
    """
    pass

def intercept_commit_user_turn(self: AgentSession):
    """
    This function is called when the user turn is committed.
    """
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self)
    )
    if session_info is None:
        return
    start_new_turn(session_info)

def pre_hook(self: AgentSession, hook_name, args, kwargs):
    try:
        if hook_name == "start":
            room = kwargs.get("room")
            room_name = kwargs.get("room_name")
            agent = kwargs.get("agent")
            get_thread_pool_executor().submit(
                intercept_session_start, self, room, room_name, agent
            )
        elif hook_name == "_update_agent_state":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(
                intercept_update_agent_state, self, args[0]
            )
        elif hook_name == "generate_reply":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(intercept_generate_reply, self, args[0])
        elif hook_name == "_update_user_state":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(
                intercept_user_state_changed, self, args[0]
            )
        elif hook_name == "emit":
            if args[0] == "metrics_collected":
                # We do not need to handle this as it is to be handled in the agent activity
                pass
            elif args[0] == "_on_metrics_collected":
                pass
            elif args[0] == "function_tools_executed":
                if not args or len(args) == 0:
                    return
                get_thread_pool_executor().submit(
                    handle_tool_call_executed, self, args[1]
                )
            elif args[0] == "agent_state_changed":
                pass
            else:
                scribe().debug(
                    f"[Internal][{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}"
                )
        elif hook_name == "end":
            pass
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self: AgentSession, result, hook_name, args, kwargs):
    try:
        if hook_name == "emit":
            if args[0] == "metrics_collected":
                pass
        elif hook_name == "_conversation_item_added":
            if args and len(args) > 0:
                item = args[0]
                if (
                    hasattr(item, "role")
                    and item.role == "assistant"
                    and hasattr(item, "content")
                    and item.content
                ):
                    content = (
                        item.content[0]
                        if isinstance(item.content, list)
                        else str(item.content)
                    )
                    get_thread_pool_executor().submit(
                        handle_agent_response_complete, self, content
                    )
        elif hook_name == "commit_user_turn":
            get_thread_pool_executor().submit(
                intercept_commit_user_turn, self
            )
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def instrument_agent_session(orig, name):
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
