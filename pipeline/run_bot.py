"""
run_bot.py
──────────
Main entry point for the e3-assessment voice agent.

Pipeline:
  Daily WebRTC mic  →  Deepgram STT  →  LLM (Groq/OR/Gemini)  →  Qwen3-TTS (megakernel)  →  Daily WebRTC speaker

Usage:
  python pipeline/run_bot.py

The bot will:
  1. Create a Daily room (or join DAILY_ROOM_URL if set in .env)
  2. Print the room URL — open it in any browser to talk
  3. Stream your speech through the full pipeline
  4. Print per-utterance latency metrics to stdout
"""

import asyncio
import os
import sys
import time

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextAggregator,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from pipeline.llm_fallback import build_llm_service
from pipeline.megakernel_tts_service import MegakernelTTSService
from pipeline.metrics_observer import MetricsObserver

SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "24000"))

SYSTEM_PROMPT = """You are a helpful, concise voice assistant.
Keep responses SHORT — 1-3 sentences maximum.
You are speaking aloud, so avoid markdown, bullet points, or lists.
Be conversational and warm."""


async def create_daily_room() -> tuple[str, str]:
    """
    Create a new ephemeral Daily room and return (room_url, token).
    Falls back to DAILY_ROOM_URL env var if set.
    """
    import aiohttp

    room_url = os.getenv("DAILY_ROOM_URL", "")
    daily_key = os.getenv("DAILY_API_KEY", "")

    if room_url:
        logger.info(f"Using existing Daily room: {room_url}")
        # Create a meeting token for the bot
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.daily.co/v1/meeting-tokens",
                headers={"Authorization": f"Bearer {daily_key}"},
                json={"properties": {"room_name": room_url.split("/")[-1], "is_owner": True}},
            ) as resp:
                data = await resp.json()
                token = data.get("token", "")
        return room_url, token

    if not daily_key:
        raise ValueError(
            "DAILY_API_KEY is required. Get a free key at https://dashboard.daily.co/"
        )

    async with aiohttp.ClientSession() as session:
        # Create room
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {daily_key}"},
            json={
                "properties": {
                    "exp": int(time.time()) + 3600,  # 1 hour expiry
                    "enable_chat": False,
                    "enable_knocking": False,
                    "start_audio_off": False,
                    "start_video_off": True,
                }
            },
        ) as resp:
            room_data = await resp.json()
            room_url = room_data["url"]

        # Create owner token for the bot
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={"Authorization": f"Bearer {daily_key}"},
            json={
                "properties": {
                    "room_name": room_url.split("/")[-1],
                    "is_owner": True,
                }
            },
        ) as resp:
            token_data = await resp.json()
            token = token_data["token"]

    return room_url, token


async def run_bot() -> None:
    # ── Room ──────────────────────────────────────────────────────────────────
    room_url, token = await create_daily_room()

    print("\n" + "=" * 60)
    print("  Voice agent ready!")
    print(f"  Open this URL in your browser to talk:")
    print(f"\n  👉  {room_url}\n")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # ── Transport ─────────────────────────────────────────────────────────────
    transport = DailyTransport(
        room_url,
        token,
        "Voice Agent",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
        ),
    )

    # ── STT ───────────────────────────────────────────────────────────────────
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not deepgram_key:
        raise ValueError(
            "DEEPGRAM_API_KEY is required. Get a free key at https://console.deepgram.com/"
        )

    stt = DeepgramSTTService(
        api_key=deepgram_key,
        audio_passthrough=True,
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = build_llm_service()

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts = MegakernelTTSService(
        voice="Chelsie",
        max_new_tokens=int(os.getenv("TTS_MAX_TOKENS", "1500")),
    )

    # ── Context / aggregators ─────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context  = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ── Metrics observer ──────────────────────────────────────────────────────
    metrics = MetricsObserver(tts_service=tts)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),          # Daily WebRTC mic → UserAudioRawFrame
        stt,                        # speech → TranscriptionFrame
        context_aggregator.user(),  # TranscriptionFrame → LLMContext message
        llm,                        # LLMContext → LLMTextFrame (streamed)
        tts,                        # LLMTextFrame → TTSAudioRawFrame (streamed)
        transport.output(),         # TTSAudioRawFrame → Daily WebRTC speaker
        context_aggregator.assistant(),  # capture assistant response
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            allow_interruptions=True,
        ),
        observers=[metrics],
    )

    # ── Greet the user when they join ─────────────────────────────────────────
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant.get('id', 'unknown')}")
        # Greet the user by injecting a synthetic assistant turn
        from pipecat.frames.frames import LLMFullResponseStartFrame, LLMFullResponseEndFrame, TextFrame
        await task.queue_frames([
            context_aggregator.user().get_context_frame(),
            LLMFullResponseStartFrame(),
            TextFrame("Hello! I'm your voice assistant. How can I help you today?"),
            LLMFullResponseEndFrame(),
        ])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left — shutting down.")
        await task.cancel()

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_bot())
