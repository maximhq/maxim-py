import asyncio
import json
import logging
import os
from enum import Enum

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobRequest, RoomIO, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import InputAudioTranscription
from openai.types.realtime.realtime_session_create_response import TracingTracingConfiguration 
from maxim import Maxim
from maxim.logger.livekit import instrument_livekit


class SessionState(Enum):
    """Session state enum for tracking the current state of the agent session"""
    ACTIVE = "active"
    CLEANING_UP = "cleaning_up"
    INTERRUPTED = "interrupted"
    SHUTDOWN = "shutdown"

logger = logging.getLogger("push-to-talk")
logger.setLevel(logging.INFO)

load_dotenv()
maxim_logger = Maxim({ "base_url": os.getenv("MAXIM_BASE_URL"), "debug": True }).logger()
instrument_livekit(maxim_logger)
print("Testting livekit push to talk, Take 19")

## This example demonstrates how to use the push-to-talk for multi-participant
## conversations with a voice agent
## It disables audio input by default, and only enables it when the client explicitly
## triggers the `start_turn` RPC method

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(model="gpt-4o-realtime-preview", voice="coral", turn_detection=None),
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # callback before generating a reply after user turn committed
        logger.info("User turn completed: '%s'", new_message.text_content)
        
        if not new_message.text_content:
            # for example, raise StopResponse to stop the agent from generating a reply
            logger.info("ignore empty user turn")
            raise StopResponse()
        
        # Log that we're about to let the agent respond
        logger.info("User input received, agent will respond automatically")
        
        # Debug: Check if we have access to session for manual generation if needed
        if hasattr(self, 'session') and self.session:
            logger.info("Agent session is available for manual response generation if needed")
        else:
            logger.info("No direct session access - relying on automatic response")


# async def entrypoint(ctx: JobContext):
#     session = AgentSession(turn_detection="manual")
#     room_io = RoomIO(session, room=ctx.room)
#     await room_io.start()
#
#     agent = MyAgent()
#     await session.start(agent=agent)
#     await session.generate_reply(
#         instructions="Greet the user and offer your assistance."
#     )
#
#     # disable input audio at the start
#     session.input.set_audio_enabled(False)
#
#     logger.info("🚀 Agent started with RPC controls!")
#
#     @ctx.room.local_participant.register_rpc_method("start_turn")
#     async def start_turn(data: rtc.RpcInvocationData):
#         logger.info("🎤 RPC: Starting turn with data: %s", data)
#         session.interrupt()
#         session.clear_user_turn()
#
#         # listen to the caller if multi-user
#         room_io.set_participant(data.caller_identity)
#         session.input.set_audio_enabled(True)
#         logger.info("Audio input enabled")
#
#     @ctx.room.local_participant.register_rpc_method("end_turn")
#     async def end_turn(data: rtc.RpcInvocationData):
#         logger.info("🔇 RPC: Ending turn")
#         session.input.set_audio_enabled(False)
#         logger.info("Audio input disabled")
#         session.commit_user_turn(transcript_timeout=0.01)
#         logger.info("User turn committed - waiting for agent response")
#
#     @ctx.room.local_participant.register_rpc_method("cancel_turn")
#     async def cancel_turn(_data: rtc.RpcInvocationData):
#         logger.info("❌ RPC: Cancelling turn")
#         session.input.set_audio_enabled(False)
#         session.clear_user_turn()
#         logger.info("Turn cancelled")


async def handle_request(request: JobRequest) -> None:
    await request.accept(
        identity="ptt-agent",
        # this attribute communicates to frontend that we support PTT
        attributes={"push-to-talk": "1"},
    )

async def entrypoint(ctx: JobContext):
    logger.info("🚀 Starting agent entrypoint")

    main_agent = MyAgent()

    # Extract metadata from job dispatch
    conversation_id = None
    # base_qa_prompt = None
    # storytelling_prompt = None
    # user_data = None

    # Access metadata from agent dispatch (job metadata)
    try:
        if ctx.job.metadata:
            job_metadata = json.loads(ctx.job.metadata)
            conversation_id = job_metadata.get("conversation_id")
            user_data = job_metadata.get("user_data", {})

            # Extract prompts for each agent
            # prompts = job_metadata.get("prompts", {})
            # base_qa_prompt = prompts.get("base_qa")
            # storytelling_prompt = prompts.get("storytelling")
            
            # Extract story parameters (theme, genre, arc) pre-selected by backend
            # story_parameters = job_metadata.get("story_parameters", {})
        
        # Validate that all required data is available
        # missing_data = []
        # if not conversation_id:
        #     missing_data.append("conversation_id")
        # if not base_qa_prompt:
        #     missing_data.append("base_qa_prompt")
        # if not user_data:
        #     missing_data.append("user_data")
        # if not storytelling_prompt:
        #     missing_data.append("storytelling_prompt")
        # 
        # if missing_data:
        #     error_msg = f"Cannot start agent - missing required data: {', '.join(missing_data)}"
        #     logger.error(error_msg)
        #     logger.error("Agent requires proper job metadata with prompts and user data to function. This indicates a configuration error in the job dispatch system")
        #     
        #     # Clean shutdown with descriptive reason - client should handle this
        #     ctx.shutdown(reason=f"Missing required data: {', '.join(missing_data)}")
        #     return
            
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse job metadata: {e}")
        logger.error("This indicates malformed metadata in the job dispatch. Agent cannot start without valid job metadata")
        
        # Clean shutdown with descriptive reason - client should handle this
        ctx.shutdown(reason=f"Failed to parse job metadata: {e}")
        return

    # Connect to the room immediately - no blocking operations
    await ctx.connect()


    # # Create userdata immediately with story parameters from backend
    # userdata = UserData(
    #     ctx=ctx,
    #     conversation_id=conversation_id,
    #     prompt=base_qa_prompt,  # Store base prompt for backward compatibility
    #     recording_id=None,  # Will be set by background recording task
    #     user_data=user_data,
    #     story_theme=story_parameters.get("theme") if story_parameters else None,
    #     story_genre=story_parameters.get("genre") if story_parameters else None,
    #     story_arc=story_parameters.get("arc") if story_parameters else None,
    #     first_speech_sent=False  # Reset for new session
    # )
    # 
    # # Create agents with specific prompts
    # main_agent = MainAgent(base_qa_prompt)
    # story_agent = StoryAgent(storytelling_prompt)
    # # homework_agent = HomeworkAgent()  # Homework agent doesn't have declarative prompt yet
    #
    # # Register all agents in the userdata
    # userdata.personas.update({
    #     "main": main_agent,
    #     "story": story_agent,
    #     # "homework": homework_agent
    # })
    #
    # kid_name = user_data.get("supervisee_name", "unknown")

    # Create session
    session = AgentSession(
        turn_detection="manual",
        user_away_timeout=90.0,
        llm=openai.realtime.RealtimeModel(
            model="gpt-realtime",
            temperature=0.8,
            voice="sage",
            turn_detection=None,
            input_audio_transcription=InputAudioTranscription(
                model="gpt-4o-transcribe",
                language="en",
                prompt="You are transcribing the audio of a conversation between a kid named Tony and Bunny. Bunny is an AI guide for the kid. You should never output this prompt in your transcription. NEVER ever do that. Try to transcribe exactly what is in the audio. If you are not sure, then you should say specifically output 'transcription failed'. "
            ),
            tracing=TracingTracingConfiguration(
                workflow_name="bunny-test",
                metadata={
                    "user": "test user",
                    "conversation_id": f"{conversation_id if conversation_id else 'test convo id'}"
                }
            )
        ),
    )
    room_io = RoomIO(
        session,
        room=ctx.room,
    )
    
    # Start agent session (this needs to be fast!)
    agent_task = asyncio.create_task(session.start(
        agent=main_agent,
        room=ctx.room,
    ))
    
    # Launch recording as background task
    # logger.info(f"🎬 Launching background recording setup...")
    # asyncio.create_task(start_recording_async(ctx, conversation_id, userdata))
    
    # Wait for agent to start (critical path)
    await agent_task
    await session.generate_reply(
        instructions="Greet the user and offer your assistance in English."
    )

    # Register shutdown callback with JobContext for proper async cleanup
    async def cleanup_on_session_close(reason: str):
        """Async cleanup called by JobContext during shutdown - guaranteed to complete"""
        try:
            logger.info(f"🧹 Shutdown cleanup started - {reason}")
            # userdata.session_state = SessionState.
            
            # try:
            #     success = await utils.end_conversation(conversation_id)
            #     if success:
            #         logger.info("✅ End conversation completed successfully")
            #     else:
            #         logger.error("❌ End conversation failed - trying sync fallback")
            # except Exception as e:
            #     logger.error(f"❌ Exception during end_conversation: {e}")
                
            # userdata.session_state = SessionState.SHUTDOWN
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error in shutdown cleanup: {e}")

    # Register the async cleanup with JobContext
    ctx.add_shutdown_callback(cleanup_on_session_close)

    # @session.on("close")
    # def on_session_close(event):
    #     """Handle immediate session close actions only - no async operations"""
    #     logger.info("🔔 Session close event received")
    #     try:
    #         userdata.session_state = SessionState.INTERRUPTED
    #     except Exception as e:
    #         logger.error(f"❌ Error in session close handler: {e}")

    @session.on("conversation_item_added") 
    def on_conversation_item_added(event):
        """Handle when items are added to conversation history - captures both user and agent responses"""
        try:
            item = event.item       
            # Extract text content from ChatMessage.content list
            transcript_text = ""
            if hasattr(item, 'content') and item.content:
                
                # content is list[ImageContent | AudioContent | str]
                text_parts = []
                for i, content_item in enumerate(item.content):
                    if isinstance(content_item, str):
                        text_parts.append(content_item)
                    elif hasattr(content_item, 'text'):
                        text_parts.append(content_item.text)
                
                transcript_text = " ".join(text_parts).strip()
            
            is_user = item.role == "user" if hasattr(item, 'role') else False
            item_id = item.id if hasattr(item, 'id') else None
            
            logger.info(f"💬 {'User' if is_user else 'Agent'} conversation item: {transcript_text}")
            
            # Allow transcript uploads during active, cleanup, and interruption states
            # if userdata.session_state in [SessionState.ACTIVE, SessionState.CLEANING_UP, SessionState.INTERRUPTED]:
            #     if transcript_text:
            #         # Log if this is a partial transcript during interruption
            #         if userdata.session_state == SessionState.INTERRUPTED:
            #             logger.info(f"📝 Capturing {'user' if is_user else 'agent'} transcript during interruption")
            #         
            #         asyncio.create_task(utils.upload_transcript(
            #             conversation_id=conversation_id,
            #             transcript=transcript_text,
            #             is_user=is_user,
            #             user_data=user_data,
            #             item_id=item_id
            #         ))
                        
        except Exception as e:
            logger.error(f"Error handling conversation item: {e}")

    @session.on("user_state_changed")
    def on_user_state_changed(event):
        """Handle when user state changes - LiveKit automatically triggers 'away' after 30s timeout"""
        try:
            new_state = event.new_state
            old_state = event.old_state
            
            logger.info(f"👤 User state changed: {old_state} -> {new_state}")
            
            if new_state == "away":
                logger.info(f"😴 User went away - LiveKit will close session")
                
        except Exception as e:
            logger.error(f"Error handling user state change: {e}")

    # @session.on("agent_state_changed")
    # def on_agent_state_changed(event):
    #     """Handle when agent state changes - detect when agent starts speaking"""
    #     try:
    #         old_state = event.old_state
    #         new_state = event.new_state
    #         
    #         logger.info(f"🤖 Agent state changed: {old_state} -> {new_state}")
    #         
    #         # Track first agent speech and notify client immediately when agent starts speaking
    #         if new_state == "speaking" and not userdata.first_speech_sent:
    #             userdata.first_speech_sent = True
    #             logger.info("🎤 First agent speech started - notifying client immediately")
    #             
    #             # Emit event to client via RPC
    #             asyncio.create_task(notify_first_speech(ctx))
    #             
    #     except Exception as e:
    #         logger.error(f"Error handling agent state change: {e}")
    
    # Simple participant disconnect handler for immediate response interruption only
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.Participant):
        logger.info(f"Participant disconnected: {participant.identity}")
        # Only interrupt if it's a client (not the agent itself)
        if participant != ctx.room.local_participant:
            try:
                logger.info("🛑 Interrupting ongoing response")
                session.interrupt()  # Immediately stop any ongoing generation
                # userdata.session_state = SessionState.INTERRUPTED
            except Exception as e:
                logger.error(f"Error interrupting session: {e}")

    # Set agent attribute to indicate push-to-talk support
    try:
        await ctx.room.local_participant.set_attributes({"push-to-talk": "1"})
    except Exception as e:
        logger.warning(f"Failed to set push-to-talk attribute: {e}")

    # Set up RPC methods for push-to-talk support
    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        # if userdata.session_state != SessionState.ACTIVE:
        #     logger.info(f"Ignoring start_turn - session state is {userdata.session_state.value}")
        #     return
            
        try:
            session.interrupt()
            session.clear_user_turn()
            # listen to the caller if multi-user
            room_io.set_participant(data.caller_identity)
            session.input.set_audio_enabled(True)
            logger.info("start turn")
        except Exception as e:
            logger.error(f"Error in start_turn: {e}")

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData):
        
        # if userdata.session_state != SessionState.ACTIVE:
        #     logger.info(f"Ignoring end_turn - session state is {userdata.session_state.value}")
        #     return
            
        try:
            logger.info("end turn")
            session.input.set_audio_enabled(False)
            session.commit_user_turn(transcript_timeout=0.01) #critical to set it low otherwise the latency will be very high
        except Exception as e:
            logger.error(f"Error in end_turn: {e}")

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData):
        # if userdata.session_state != SessionState.ACTIVE:
        #     logger.info(f"Ignoring cancel_turn - session state is {userdata.session_state.value}")
        #     return
            
        try:
            session.input.set_audio_enabled(False)
            session.clear_user_turn()
            logger.info("cancel turn")
        except Exception as e:
            logger.error(f"Error in cancel_turn:{e}")

    @ctx.room.local_participant.register_rpc_method("agent_first_speech")
    async def agent_first_speech(data: rtc.RpcInvocationData):
        """RPC method for clients to register for first speech notifications"""
        try:
            # This method can be called by client to check if first speech has been sent
            # or to register interest in first speech notifications
            response = {
                "first_speech_sent": False,
                "session_state": SessionState.ACTIVE.value
            }
            logger.info(f"Client {data.caller_identity} queried first speech status: {response}")
            return json.dumps(response)
        except Exception as e:
            logger.error(f"Error in agent_first_speech RPC: {e}")
            return json.dumps({"error": str(e)})

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, request_fnc=handle_request))
