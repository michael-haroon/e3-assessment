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
from typing import AsyncGenerator, Optional

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


class MegakernelTTSService(TTSService):
    """
    Pipecat TTSService that routes synthesis through Qwen3-TTS with the
    AlpinDale megakernel as the talker LM backend.

    Parameters
    ──────────
    voice:           CosyVoice speaker ID (default "Chelsie")
    max_new_tokens:  hard cap on codec tokens per utterance
    """

    def __init__(
        self,
        *,
        voice: str = "ryan",
        max_new_tokens: int = 1500,
        **kwargs,
    ):
        # Pass sample_rate to base class so Pipecat knows what rate we emit
        super().__init__(
            sample_rate=TTS_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )

        self._tts_pipeline = Qwen3TTSPipeline(
            speaker=voice,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )

        # Latency tracking (accessible to the benchmark layer)
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

                self._last_audio_bytes += len(pcm_bytes)

                yield TTSAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=TTS_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )

            # Capture megakernel stats from the last synthesis run
            stats = getattr(self._tts_pipeline, "last_mk_stats", {})
            self.last_tok_per_s = stats.get("tok_per_s", 0.0)

            # Compute RTF from audio bytes we've seen
            n_samples = self._last_audio_bytes // 2
            audio_dur = n_samples / TTS_SAMPLE_RATE if TTS_SAMPLE_RATE else 0
            self.last_rtf = (
                (time.perf_counter() - ttfc_start) / audio_dur
                if audio_dur > 0 else 0.0
            )

            logger.debug(
                f"TTS done: TTFC={self.last_ttfc_ms:.0f}ms "
                f"tok/s={self.last_tok_per_s:.0f}"
            )

        except Exception as exc:
            logger.error(f"MegakernelTTS error: {exc!r}")
            yield ErrorFrame(error=str(exc), exception=exc)
