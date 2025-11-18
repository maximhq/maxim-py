"""
Context Summarization with Realtime API

Build an end-to-end voice bot that listens to your mic, speaks back in real time
and summarises long conversations so quality never drops.

Prerequisites:
- Python >= 3.10
- OpenAI API key (set OPENAI_API_KEY environment variable)
- Mic + speakers (grant OS permission if prompted)
"""

# Standard library imports
import os
import sys
import json
import base64
from dataclasses import dataclass, field
from typing import List, Literal

# Third-party imports
import asyncio
import sounddevice as sd  # microphone capture
import simpleaudio  # speaker playback
import websockets  # WebSocket client
import openai  # OpenAI Python SDK >= 1.14.0
import dotenv

dotenv.load_dotenv()

# Set your API key safely
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found – please set env var or edit this cell.")

# Audio/config knobs
SAMPLE_RATE_HZ = 24_000  # Required by pcm16
CHUNK_DURATION_MS = 40  # chunk size for audio capture
BYTES_PER_SAMPLE = 2  # pcm16 = 2 bytes/sample
SUMMARY_TRIGGER = 2_000  # Summarise when context >= this
KEEP_LAST_TURNS = 2  # Keep these turns verbatim
SUMMARY_MODEL = "gpt-4o-mini"  # Cheaper, fast summariser


@dataclass
class Turn:
    """One utterance in the dialogue (user **or** assistant)."""

    role: Literal["user", "assistant"]
    item_id: str  # Server‑assigned identifier
    text: str | None = None  # Filled once transcript is ready


@dataclass
class ConversationState:
    """All mutable data the session needs — nothing more, nothing less."""

    history: List[Turn] = field(default_factory=list)  # Ordered log
    waiting: dict[str, asyncio.Future] = field(
        default_factory=dict
    )  # Pending transcript fetches
    summary_count: int = 0

    latest_tokens: int = 0  # Window size after last reply
    summarising: bool = False  # Guard so we don't run two summaries at once


def print_history(state) -> None:
    """Pretty-print the running transcript so far."""
    print("—— Conversation so far ———————————————")
    for turn in state.history:
        text_preview = (turn.text or "").strip().replace("\n", " ")
        print(f"[{turn.role:<9}] {text_preview}  ({turn.item_id})")
    print("——————————————————————————————————————————")


async def mic_to_queue(pcm_queue: asyncio.Queue[bytes]) -> None:
    """
    Capture raw PCM‑16 microphone audio and push ~CHUNK_DURATION_MS chunks
    to *pcm_queue* until the surrounding task is cancelled.

    Parameters
    ----------
    pcm_queue : asyncio.Queue[bytes]
        Destination queue for PCM‑16 frames (little‑endian int16).
    """
    blocksize = int(SAMPLE_RATE_HZ * CHUNK_DURATION_MS / 1000)

    def _callback(indata, _frames, _time, status):
        if status:  # XRuns, device changes, etc.
            print("⚠️", status, file=sys.stderr)
        try:
            pcm_queue.put_nowait(bytes(indata))  # 1‑shot enqueue
        except asyncio.QueueFull:
            # Drop frame if upstream (WebSocket) can't keep up.
            pass

    # RawInputStream is synchronous; wrap in context manager to auto‑close.
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE_HZ,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        try:
            # Keep coroutine alive until cancelled by caller.
            await asyncio.Event().wait()
        finally:
            print("⏹️  Mic stream closed.")


# Helper function to encode audio chunks in base64
def b64(blob):
    return base64.b64encode(blob).decode()


async def queue_to_websocket(pcm_queue: asyncio.Queue[bytes], ws):
    """Read audio chunks from queue and send as JSON events."""
    try:
        while (chunk := await pcm_queue.get()) is not None:
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": b64(chunk),
                    }
                )
            )
    except websockets.ConnectionClosed:
        print("WebSocket closed – stopping uploader")


async def run_summary_llm(text: str) -> str:
    """Call a lightweight model to summarise `text`."""
    resp = await asyncio.to_thread(
        lambda: openai.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise in French the following conversation "
                    "in one concise paragraph so it can be used as "
                    "context for future dialogue.",
                },
                {"role": "user", "content": text},
            ],
        )
    )
    return resp.choices[0].message.content.strip()


async def summarise_and_prune(ws, state):
    """Summarise old turns, delete them server‑side, and prepend a single summary
    turn locally + remotely."""
    state.summarising = True
    print(
        f"⚠️  Token window ≈{state.latest_tokens} ≥ {SUMMARY_TRIGGER}. Summarising…",
    )
    old_turns, recent_turns = (
        state.history[:-KEEP_LAST_TURNS],
        state.history[-KEEP_LAST_TURNS:],
    )
    convo_text = "\n".join(f"{t.role}: {t.text}" for t in old_turns if t.text)

    if not convo_text:
        print("Nothing to summarise (transcripts still pending).")
        state.summarising = False
        return

    summary_text = await run_summary_llm(convo_text) if convo_text else ""
    state.summary_count += 1
    summary_id = f"sum_{state.summary_count:03d}"
    state.history[:] = [Turn("assistant", summary_id, summary_text)] + recent_turns

    print_history(state)

    # Create summary on server
    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "previous_item_id": "root",
                "item": {
                    "id": summary_id,
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": summary_text}],
                },
            }
        )
    )

    # Delete old items
    for turn in old_turns:
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.delete",
                    "item_id": turn.item_id,
                }
            )
        )

    print(f"✅ Summary inserted ({summary_id})")

    state.summarising = False


async def fetch_full_item(
    ws, item_id: str, state: ConversationState, attempts: int = 1
):
    """
    Ask the server for a full conversation item; retry up to 5× if the
    transcript field is still null.  Resolve the waiting future when done.
    """
    # If there is already a pending fetch, just await it
    if item_id in state.waiting:
        return await state.waiting[item_id]

    fut = asyncio.get_running_loop().create_future()
    state.waiting[item_id] = fut

    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.retrieve",
                "item_id": item_id,
            }
        )
    )
    item = await fut

    # If transcript still missing retry (max 5×)
    if attempts < 5 and not item.get("content", [{}])[0].get("transcript"):
        await asyncio.sleep(0.4 * attempts)
        return await fetch_full_item(ws, item_id, state, attempts + 1)

    # Done – remove the marker
    state.waiting.pop(item_id, None)
    return item


# --------------------------------------------------------------------------- #
# Realtime session                                                          #
# --------------------------------------------------------------------------- #
async def realtime_session(model="gpt-realtime", voice="shimmer", enable_playback=True):
    """
    Main coroutine: connects to the Realtime endpoint, spawns helper tasks,
    and processes incoming events in a big async‑for loop.
    """
    state = ConversationState()  # Reset state for each run

    pcm_queue: asyncio.Queue[bytes] = asyncio.Queue()
    assistant_audio: List[bytes] = []

    # ----------------------------------------------------------------------- #
    # Open the WebSocket connection to the Realtime API                       #
    # ----------------------------------------------------------------------- #
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {openai.api_key}"}

    async with websockets.connect(url, additional_headers=headers, max_size=1 << 24) as ws:
        # ------------------------------------------------------------------- #
        # Wait until server sends session.created                             #
        # ------------------------------------------------------------------- #
        while json.loads(await ws.recv())["type"] != "session.created":
            pass
        print("session.created ✅")

        # ------------------------------------------------------------------- #
        # Configure session: voice, modalities, audio formats, transcription  #
        # ------------------------------------------------------------------- #
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "model": model,  # Fixed: was using variable name instead of string key
                        "voice": voice,
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "gpt-4o-transcribe"},
                    },
                }
            )
        )

        # ------------------------------------------------------------------- #
        # Launch background tasks: mic capture → queue → websocket            #
        # ------------------------------------------------------------------- #
        mic_task = asyncio.create_task(mic_to_queue(pcm_queue))
        upl_task = asyncio.create_task(queue_to_websocket(pcm_queue, ws))

        print("🎙️ Speak now (Ctrl‑C to quit)…")

        try:
            # ------------------------------------------------------------------- #
            # Main event loop: process incoming events from the websocket         #
            # ------------------------------------------------------------------- #
            async for event_raw in ws:
                event = json.loads(event_raw)
                etype = event["type"]

                # --------------------------------------------------------------- #
                # User just spoke ⇢ conversation.item.created (role = user)        #
                # --------------------------------------------------------------- #
                if (
                    etype == "conversation.item.created"
                    and event["item"]["role"] == "user"
                ):
                    item = event["item"]
                    text = None
                    if item["content"]:
                        text = item["content"][0].get("transcript")

                    state.history.append(Turn("user", event["item"]["id"], text))

                    # If transcript not yet available, fetch it later
                    if text is None:
                        asyncio.create_task(fetch_full_item(ws, item["id"], state))

                # --------------------------------------------------------------- #
                # Transcript fetched ⇢ conversation.item.retrieved                 #
                # --------------------------------------------------------------- #
                elif etype == "conversation.item.retrieved":
                    item = event["item"]
                    content = item["content"][0] if item.get("content") else {}
                    # Fill missing transcript in history
                    for t in state.history:
                        if t.item_id == item["id"]:
                            t.text = content.get("transcript")
                            break

                    # Resolve the future if one exists
                    item_id = item["id"]
                    if item_id in state.waiting:
                        state.waiting[item_id].set_result(item)
                        state.waiting.pop(item_id, None)

                # --------------------------------------------------------------- #
                # Assistant audio arrives in deltas                               #
                # --------------------------------------------------------------- #
                elif etype == "response.output_audio.delta":
                    assistant_audio.append(base64.b64decode(event["delta"]))

                # --------------------------------------------------------------- #
                # Assistant reply finished ⇢ response.done                        #
                # --------------------------------------------------------------- #
                elif etype == "response.done":
                    for item in event["response"]["output"]:
                        if item["role"] == "assistant":
                            txt = item["content"][0]["transcript"]
                            state.history.append(Turn("assistant", item["id"], txt))
                            # print(f"\n🤖 {txt}\n")
                    state.latest_tokens = event["response"]["usage"]["total_tokens"]
                    print(
                        f"—— response.done  (window ≈{state.latest_tokens} tokens) ——"
                    )
                    print_history(state)

                    # Fetch any still‑missing user transcripts
                    for turn in state.history:
                        if (
                            turn.role == "user"
                            and turn.text is None
                            and turn.item_id not in state.waiting
                        ):
                            asyncio.create_task(
                                fetch_full_item(ws, turn.item_id, state)
                            )

                    # Playback collected audio once reply completes
                    if enable_playback and assistant_audio:
                        simpleaudio.play_buffer(
                            b"".join(assistant_audio),
                            1,
                            BYTES_PER_SAMPLE,
                            SAMPLE_RATE_HZ,
                        )
                        assistant_audio.clear()

                    # Summarise if context too large – fire in background so we don't block dialogue
                    if (
                        state.latest_tokens >= SUMMARY_TRIGGER
                        and len(state.history) > KEEP_LAST_TURNS
                        and not state.summarising
                    ):
                        asyncio.create_task(summarise_and_prune(ws, state))

        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            mic_task.cancel()
            await pcm_queue.put(None)
            await upl_task


def main():
    """Run the realtime session."""
    asyncio.run(realtime_session())


if __name__ == "__main__":
    main()
