import logging
import os
import uuid
import dotenv
from livekit import agents
from livekit import api as livekit_api
from livekit.agents import Agent, AgentSession, function_tool
from livekit.protocol.room import CreateRoomRequest
from livekit.plugins import google
from maxim import Maxim
from maxim.logger.livekit import instrument_livekit
from tavily import TavilyClient

# Load environment variables
dotenv.load_dotenv(override=True)
logging.basicConfig(level=logging.DEBUG)

logger = Maxim({ "base_url": os.getenv("MAXIM_BASE_URL") }).logger()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Maxim event instrumentation
def on_event(event: str, data: dict):
    if event == "maxim.trace.started":
        trace_id = data["trace_id"]
        trace = data["trace"]
        logging.debug(f"Trace started - ID: {trace_id}", extra={"trace": trace})
    elif event == "maxim.trace.ended":
        trace_id = data["trace_id"]
        trace = data["trace"]
        logging.debug(f"Trace ended - ID: {trace_id}", extra={"trace": trace})

instrument_livekit(logger, on_event)

class InterviewAgent(Agent):
    def __init__(self, jd: str) -> None:
        super().__init__(instructions=f"You are a professional interviewer. The job description is: {jd}\nAsk relevant interview questions, listen to answers, and follow up as a real interviewer would.")

    @function_tool()
    async def web_search(self, query: str) -> str:
        """
        Performs a web search for the given query.
        """
        if not TAVILY_API_KEY:
            return "Tavily API key is not set. Please set the TAVILY_API_KEY environment variable."
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        try:
            response = tavily_client.search(query=query, search_depth="basic")
            if response.get('answer'):
                return response['answer']
            return str(response.get('results', 'No results found.'))
        except Exception as e:
            return f"An error occurred during web search: {e}"

async def entrypoint(ctx: agents.JobContext):
    # Prompt user for JD at the start
    # jd = input("Paste the Job Description (JD) and press Enter:\n")
    jd = """
    Job Title: Audio Engineer
About the Role

We are looking for a talented and detail-oriented Audio Engineer to join our team. The ideal candidate will be responsible for recording, mixing, editing, and mastering high-quality audio for various projects. You’ll work closely with producers, musicians, voice-over artists, and content creators to ensure sound quality meets professional standards and enhances the overall experience.

Key Responsibilities

Set up, operate, and maintain audio recording equipment in studios or live environments

Record, edit, mix, and master audio tracks for music, film, podcasts, video games, and other media

Ensure high-quality sound by managing levels, EQ, effects, and noise reduction

Collaborate with producers, artists, and creative teams to achieve the desired audio aesthetic

Troubleshoot and resolve technical issues related to audio equipment and software

Maintain a library of audio assets and ensure proper file organization

Stay updated on the latest audio technologies, plugins, and best practices

Qualifications

Proven experience as an audio engineer, sound designer, or similar role

Proficiency with Digital Audio Workstations (DAWs) such as Pro Tools, Logic Pro, Ableton Live, Cubase, or Reaper

Strong knowledge of audio recording, editing, and mixing techniques

Familiarity with microphones, audio interfaces, mixing consoles, and studio monitors

Understanding of acoustics and soundproofing techniques

Ability to work under tight deadlines and manage multiple projects

Strong attention to detail and a creative ear for sound

Preferred Skills

Experience with live sound engineering and stage setups

Knowledge of audio post-production for film, TV, or gaming

Familiarity with spatial/3D audio and immersive sound technologies

Basic knowledge of music theory or sound design is a plus

What We Offer

Competitive salary and project-based opportunities

Access to state-of-the-art audio equipment and studio facilities

Collaborative and creative work environment

Opportunities to work on diverse and high-impact projects
    """
    room_name = os.getenv("LIVEKIT_ROOM_NAME") or f"interview-room-{uuid.uuid4().hex}"
    lkapi = livekit_api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    try:
        req = CreateRoomRequest(
            name=room_name,
            empty_timeout=600,        # keep the room alive 10m after empty
            max_participants=2,       # interviewer + candidate
        )
        room = await lkapi.room.create_room(req)
        print(f"Room created: {room}")
        session = AgentSession(
            llm=google.beta.realtime.RealtimeModel(model="gemini-2.0-flash-exp", voice="Puck"),
        )
        await session.start(room=room, agent=InterviewAgent(jd))
        await ctx.connect()
        await session.generate_reply(
            instructions="Greet the candidate and start the interview."
        )
    finally:
        await lkapi.aclose()

if __name__ == "__main__":
    opts = agents.WorkerOptions(entrypoint_fnc=entrypoint)
    agents.cli.run_app(opts)
