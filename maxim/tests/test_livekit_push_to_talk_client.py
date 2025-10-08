import asyncio
import logging
from livekit import rtc
from pynput import keyboard
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("push-to-talk-client")

class PushToTalkClient:
    def __init__(self, room_name="agent-room", participant_name="push-to-talk-user", agent_identity="agent"):
        self.room_name = room_name
        self.participant_name = participant_name
        self.agent_identity = agent_identity  # Identity of the agent to send RPCs to
        self.room = None
        self.local_participant = None
        self.is_talking = False
        self.loop = None  # Store event loop reference
        
    def _generate_participant_token(self):
        """Generate a participant token for LiveKit"""
        try:
            from livekit import api
            
            api_key = os.getenv("LIVEKIT_API_KEY")
            api_secret = os.getenv("LIVEKIT_API_SECRET")
            
            if not api_key or not api_secret:
                logger.error("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env file")
                return None
            
            # Generate proper LiveKit token
            token = api.AccessToken(api_key, api_secret) \
                .with_identity(self.participant_name) \
                .with_name(self.participant_name) \
                .with_grants(api.VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish=True,
                    can_subscribe=True
                )).to_jwt()
            
            logger.info("Generated token for room: %s, participant: %s", self.room_name, self.participant_name)
            return token
            
        except Exception as e:
            logger.error("Failed to generate token: %s", e)
            return None
        
    async def connect(self):
        """Connect to the LiveKit room"""
        try:
            # Create room instance
            self.room = rtc.Room()
            
            # Get connection details from environment
            url = os.getenv("LIVEKIT_URL")
            if not url:
                logger.error("LIVEKIT_URL environment variable not set")
                return False
                
            # For development, you might need to generate a proper token
            # This depends on your LiveKit setup
            token = self._generate_participant_token()
            if not token:
                logger.error("Failed to generate participant token")
                return False
            
            logger.info("Connecting to %s, room: %s", url, self.room_name)
            
            # Connect to the room
            await self.room.connect(url, token)
            logger.info("Connected to room %s as %s", self.room_name, self.participant_name)
            
            self.local_participant = self.room.local_participant
            
            # Set up room event handlers
            self.room.on("participant_connected", self.on_participant_connected)
            self.room.on("participant_disconnected", self.on_participant_disconnected)
            self.room.on("connection_quality_changed", self.on_connection_quality_changed)
            
            return True
            
        except Exception as e:
            logger.error("Failed to connect to room: %s", e)
            return False
    
    def on_participant_connected(self, participant):
        logger.info("Participant connected: %s", participant.identity)
    
    def on_participant_disconnected(self, participant):
        logger.info("Participant disconnected: %s", participant.identity)
        
    def on_connection_quality_changed(self, quality):
        logger.info("Connection quality changed: %s", quality)
    
    async def start_turn(self):
        """Invoke the start_turn RPC method"""
        if self.is_talking:
            return
            
        try:
            if self.local_participant:
                self.is_talking = True
                await self.local_participant.perform_rpc(
                    destination_identity=self.agent_identity,
                    method="start_turn",
                    payload=""
                )
                logger.info("🎤 Started turn - agent is listening")
            else:
                logger.error("Not connected to room")
        except Exception as e:
            logger.error("Failed to start turn: %s", e)
            self.is_talking = False
    
    async def end_turn(self):
        """Invoke the end_turn RPC method"""
        if not self.is_talking:
            return
            
        try:
            if self.local_participant:
                self.is_talking = False
                await self.local_participant.perform_rpc(
                    destination_identity=self.agent_identity,
                    method="end_turn",
                    payload=""
                )
                logger.info("✅ Ended turn - agent will respond")
            else:
                logger.error("Not connected to room")
        except Exception as e:
            logger.error("Failed to end turn: %s", e)
    
    async def cancel_turn(self):
        """Invoke the cancel_turn RPC method"""
        try:
            if self.local_participant:
                self.is_talking = False
                await self.local_participant.perform_rpc(
                    destination_identity=self.agent_identity,
                    method="cancel_turn",
                    payload=""
                )
                logger.info("❌ Cancelled turn")
            else:
                logger.error("Not connected to room")
        except Exception as e:
            logger.error("Failed to cancel turn: %s", e)
    
    def on_press(self, key):
        """Handle key press events"""
        try:
            if hasattr(key, 'char') and key.char:
                if key.char.lower() == 's' and not self.is_talking:
                    if self.loop and not self.loop.is_closed():
                        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.start_turn()))
                elif key.char.lower() == 'e' and self.is_talking:
                    if self.loop and not self.loop.is_closed():
                        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.end_turn()))
                elif key.char.lower() == 'c':
                    if self.loop and not self.loop.is_closed():
                        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.cancel_turn()))
                elif key.char.lower() == 'q':
                    # Stop listener
                    return False
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release events"""
        # No longer needed since we handle everything on key press
        pass
    
    def setup_keybindings(self):
        """Set up global keybindings using pynput"""
        # Collect events until released
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        
        print("\n" + "="*50)
        print("Push-to-Talk Client Started")
        print("="*50)
        print("Keybindings:")
        print("  - Press S to start turn")
        print("  - Press E to end turn and get response")
        print("  - Press C to cancel current turn")
        print("  - Press Q to quit")
        print("="*50)
        print("NOTE: On macOS, you may need to grant accessibility permissions")
        print("to this terminal app in System Preferences > Security & Privacy > Accessibility")
        print("="*50 + "\n")
    
    async def manual_input_mode(self):
        """Fallback manual input mode when keyboard monitoring fails"""
        print("\n" + "="*50)
        print("Manual Input Mode")
        print("="*50)
        print("Commands:")
        print("  s - Start turn")
        print("  e - End turn")
        print("  c - Cancel turn")
        print("  q - Quit")
        print("="*50 + "\n")
        
        while True:
            try:
                cmd = input("Enter command (s/e/c/q): ").strip().lower()
                if cmd == 's':
                    await self.start_turn()
                elif cmd == 'e':
                    await self.end_turn()
                elif cmd == 'c':
                    await self.cancel_turn()
                elif cmd == 'q':
                    break
                else:
                    print("Invalid command. Use s/e/c/q")
            except (EOFError, KeyboardInterrupt):
                break
        
        self.cleanup_and_exit()
    
    async def run(self):
        """Main execution loop"""
        self.loop = asyncio.get_running_loop()  # Store loop reference
        
        print("Connecting to LiveKit room...")
        if not await self.connect():
            logger.error("Failed to connect to LiveKit room. Please check your configuration.")
            return
        
        try:
            self.setup_keybindings()
            
            # Keep the client running
            while self.listener.running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning("Keyboard monitoring failed (likely accessibility permissions): %s", e)
            logger.info("Falling back to manual input mode...")
            await self.manual_input_mode()
        except KeyboardInterrupt:
            self.cleanup_and_exit()
    
    def cleanup_and_exit(self):
        """Clean up resources and exit"""
        print("\nShutting down...")
        if self.room:
            asyncio.create_task(self.room.disconnect())
        if hasattr(self, 'listener'):
            self.listener.stop()
        sys.exit(0)


async def main():
    # You can customize the room name here - make sure it matches your agent
    room_name = "agent-room"  # Change this to match your agent's room
    agent_identity = "agent"  # Change this to match your agent's identity
    
    client = PushToTalkClient(room_name=room_name, agent_identity=agent_identity)
    await client.run()


if __name__ == "__main__":
    # Install required package for local testing: uv sync --group dev --group local-testing
    
    # Check if pynput module is available
    try:
        from pynput import keyboard as _  # noqa: F401
    except ImportError:
        print("Please install the 'pynput' package for local testing:")
        print("  uv sync --group dev --group local-testing")
        print("  or: pip install pynput")
        sys.exit(1)
    
    # Check if required environment variables are set
    required_env_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease make sure your .env file contains:")
        print("LIVEKIT_URL=wss://your-livekit-server.com")
        print("LIVEKIT_API_KEY=your_api_key")
        print("LIVEKIT_API_SECRET=your_api_secret")
        sys.exit(1)
    
    asyncio.run(main())