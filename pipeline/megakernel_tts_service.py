"""
megakernel_tts_service.py
─────────────────────────
Custom Pipecat TTSService backed by Qwen3-TTS + megakernel talker.

Streaming contract
──────────────────
run_tts() is an async generator.  For each chunk yielded by the TTS
pipeline it immediately yields a TTSAudioRawFrame into the Pipecat
pipeline — no buffering of the full utterance.  Pipecat routes each frame
to the transport as it arrives, giving the user real-time audio.

Frame sequence per utterance:
  TTSStartedFrame   (pushed by base class when push_start_frame=True)
  TTSAudioRawFrame  × N   (one per ~150ms audio chunk)
  TTSStoppedFrame   (pushed by base class when push_stop_frames=True)
"""

import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService, TTSSettings

from tts.qwen3_tts_pipeline import Qwen3TTSPipeline, TTS_SAMPLE_RATE


# ── Settings ──────────────────────────────────────────────────────────────────
# Pipecat requires all three fields to be present (None = unsupported).
# Without this the validate_complete() warning fires on every startup.
@dataclass
class MegakernelTTSSettings(TTSSettings):
    model:    Optional[str] = None      # not configurable — fixed to Qwen3-TTS
    voice:    Optional[str] = "ryan"
    language: Optional[str] = None      # not configurable — English only


# ── Silence trimming ──────────────────────────────────────────────────────────
_SILENCE_THRESHOLD = 0.002   # RMS below this = silence
_TRIM_CHUNK_MS     = 50      # inspect in 50ms windows from the end


def _trim_trailing_silence(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """
    Remove silent/hiss frames from the end of a PCM audio buffer.
    Qwen3-TTS pads to max_new_tokens when EOS is not found — this strips
    the resulting silence/noise tail so the browser doesn't play it.
    """
    if not pcm_bytes:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_n = int(sample_rate * _TRIM_CHUNK_MS / 1000)

    # Walk backward in chunks; stop when we find non-silent audio
    trim_at = len(samples)
    for end in range(len(samples), 0, -chunk_n):
        start = max(0, end - chunk_n)
        rms = float(np.sqrt(np.mean(samples[start:end] ** 2)))
        if rms > _SILENCE_THRESHOLD:
            trim_at = end
            break

    trimmed = samples[:trim_at]
    trimmed_bytes = (trimmed * 32768.0).astype(np.int16).tobytes()

    trimmed_secs = (len(samples) - trim_at) / sample_rate
    if trimmed_secs > 0.1:
        logger.debug(f"TTS: trimmed {trimmed_secs:.2f}s of trailing silence")

    return trimmed_bytes


# ── Service ───────────────────────────────────────────────────────────────────
class MegakernelTTSService(TTSService):
    """
    Pipecat TTSService that routes synthesis through Qwen3-TTS with the
    AlpinDale megakernel as the talker LM backend.

    Parameters
    ──────────
    voice:           CosyVoice speaker ID (default "ryan")
    max_new_tokens:  hard cap on codec tokens per utterance.
                     40 ≈ 3.3s speech @ 12Hz codec rate — keeps chunks short and
                     prevents the model padding to a fixed length.
    trim_silence:    strip trailing silence/hiss from each chunk (default True)
    """

    def __init__(
        self,
        *,
        voice: str = "ryan",
        max_new_tokens: int = 40,
        trim_silence: bool = True,
        **kwargs,
    ):
        # Initialise settings so pipecat's validate_complete() is satisfied
        settings = MegakernelTTSSettings(voice=voice)

        super().__init__(
            sample_rate=TTS_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            settings=settings,
            **kwargs,
        )

        self._tts_pipeline = Qwen3TTSPipeline(
            speaker=voice,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )
        self._trim_silence = trim_silence

        # Latency tracking (accessible to the benchmark / metrics layer)
        self.last_ttfc_ms:   float = 0.0
        self.last_rtf:       float = 0.0
        self.last_tok_per_s: float = 0.0
        self._last_audio_bytes: int = 0

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """
        Core streaming synthesis method.

        Yields TTSAudioRawFrame chunks as soon as each ~150ms audio slice
        is ready.  The base class wraps this with TTSStartedFrame /
        TTSStoppedFrame.
        """
        if not text or not text.strip():
            return

        logger.debug(f"MegakernelTTS synthesizing [{text[:60]}…] ctx={context_id}")

        ttfc_start   = time.perf_counter()
        first_chunk  = True
        self._last_audio_bytes = 0
        chunks = []   # collect for optional silence trim

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            async for pcm_bytes in self._tts_pipeline.synthesize_streaming(text):
                if not pcm_bytes:
                    continue

                if first_chunk:
                    ttfc_ms = (time.perf_counter() - ttfc_start) * 1000
                    self.last_ttfc_ms = ttfc_ms
                    await self.stop_ttfb_metrics()
                    logger.info(f"TTS TTFC: {ttfc_ms:.1f} ms")
                    first_chunk = False

                chunks.append(pcm_bytes)

            # ── Silence trim ─────────────────────────────────────────────────
            # Concatenate all chunks, strip the silent tail, then re-split
            # into ~150ms frames for smooth streaming to the transport.
            if chunks:
                full_audio = b"".join(chunks)

                if self._trim_silence:
                    full_audio = _trim_trailing_silence(full_audio, TTS_SAMPLE_RATE)

                self._last_audio_bytes = len(full_audio)

                # Re-chunk into ~150ms frames (~4800 samples @ 16kHz, 2 bytes each)
                frame_bytes = int(TTS_SAMPLE_RATE * 0.15) * 2
                for offset in range(0, len(full_audio), frame_bytes):
                    chunk = full_audio[offset: offset + frame_bytes]
                    if chunk:
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=TTS_SAMPLE_RATE,
                            num_channels=1,
                            context_id=context_id,
                        )

            # ── Stats ─────────────────────────────────────────────────────────
            stats = getattr(self._tts_pipeline, "last_mk_stats", {})
            self.last_tok_per_s = stats.get("tok_per_s", 0.0)

            n_samples  = self._last_audio_bytes // 2
            audio_dur  = n_samples / TTS_SAMPLE_RATE if TTS_SAMPLE_RATE else 0
            wall_time  = time.perf_counter() - ttfc_start
            self.last_rtf = wall_time / audio_dur if audio_dur > 0 else 0.0

            logger.debug(
                f"TTS done: TTFC={self.last_ttfc_ms:.0f}ms "
                f"tok/s={self.last_tok_per_s:.0f} "
                f"audio={audio_dur:.2f}s RTF={self.last_rtf:.3f}"
            )

        except Exception as exc:
            logger.error(f"MegakernelTTS error: {exc!r}")
            yield ErrorFrame(error=str(exc), exception=exc)
