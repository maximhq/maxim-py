import asyncio
import logging
import os
import select
import sys
import termios
import tty
import threading
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, RoomIO, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.plugins import openai
from maxim import Maxim, scribe
from maxim.logger.livekit import instrument_livekit

logger = logging.getLogger("push-to-talk")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates push-to-talk with terminal key bindings
## No RPC calls needed - use keyboard shortcuts directly in the terminal:
## - 's': Start turn (enable microphone)
## - 'e': End turn (disable microphone and process)
## - 'c': Cancel current turn
## - 'q': Quit the application


logger = Maxim({ "base_url": os.getenv("MAXIM_BASE_URL") }).logger()
instrument_livekit(logger)


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(voice="alloy", turn_detection=None),
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # callback before generating a reply after user turn committed
        scribe.scribe().info(f"on_user_turn_completed: {new_message.text_content}")
        if not new_message.text_content:
            # for example, raise StopResponse to stop the agent from generating a reply
            scribe.scribe().info("ignore empty user turn")
            raise StopResponse()

        # Let the agent automatically generate a response
        scribe.scribe().info("User turn completed - agent will respond automatically")


class KeyboardHandler:
    def __init__(self, session: AgentSession):
        self.session = session
        self.running = True
        self.old_settings = None
        
    async def start_turn(self):
        """Start a new turn - enable audio input"""
        scribe.scribe().info("🎤 Starting turn - speak now!")
        self.session.interrupt()
        self.session.clear_user_turn()
        self.session.input.set_audio_enabled(True)
            
    async def end_turn(self):
        """End current turn - disable audio and commit"""
        scribe.scribe().info("🔇 Ending turn - processing...")
        self.session.input.set_audio_enabled(False)
        self.session.commit_user_turn()
            
    async def cancel_turn(self):
        """Cancel current turn - disable audio and clear"""
        scribe.scribe().info("❌ Cancelling turn")
        self.session.input.set_audio_enabled(False)
        self.session.clear_user_turn()
            
    def quit_app(self):
        """Quit the application"""
        scribe.scribe().info("👋 Quitting application...")
        self.running = False
        
    def setup_terminal(self):
        """Setup terminal for raw input"""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
    def restore_terminal(self):
        """Restore terminal settings"""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
    async def keyboard_listener(self):
        """Listen for keyboard input in a separate thread"""
        self.setup_terminal()
        
        try:
            print("\n" + "="*50)
            print("🎤 PUSH-TO-TALK CONTROLS:")
            print("  s: Start turn (enable microphone)")
            print("  e: End turn (disable microphone and process)")
            print("  c: Cancel current turn")
            print("  q: Quit application")
            print("="*50 + "\n")
            
            while self.running:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    
                    if char == 's':  # Start turn
                        await self.start_turn()
                    elif char == 'e':  # End turn
                        await self.end_turn()
                    elif char == 'c':  # Cancel
                        await self.cancel_turn()
                    elif char == 'q':  # Quit
                        self.quit_app()
                        break
                        
        except KeyboardInterrupt:
            scribe.scribe().info("Keyboard interrupt received")
            self.quit_app()
        finally:
            self.restore_terminal()


async def entrypoint(ctx: JobContext):
    session = AgentSession(turn_detection="manual")
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()

    agent = MyAgent()
    await session.start(agent=agent)

    # disable input audio at the start
    session.input.set_audio_enabled(False)
    
    # Setup keyboard handler
    keyboard_handler = KeyboardHandler(session)
    
    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=keyboard_handler.keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    scribe.scribe().info("🚀 Agent started! Use keyboard controls to interact.")
    
    # Keep the main thread alive
    try:
        while keyboard_handler.running:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        scribe.scribe().info("Shutting down...")
    finally:
        keyboard_handler.running = False
        keyboard_handler.restore_terminal()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))