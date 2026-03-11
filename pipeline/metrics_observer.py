"""
metrics_observer.py
───────────────────
Pipecat pipeline observer that collects and prints per-utterance
latency metrics to stdout.

Metrics reported
────────────────
  tok/s     — megakernel decode throughput
  TTFC      — time from LLM first token to first audio chunk leaving TTS
  RTF       — synthesis_time / audio_duration  (< 0.15 is the target)
  E2E       — user stops speaking → first audio out
"""

import time
from typing import Optional

from loguru import logger
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStoppedSpeakingFrame,
)

from pipeline.megakernel_tts_service import MegakernelTTSService


class MetricsObserver(BaseObserver):
    """
    Observes pipeline frames and emits a metrics summary line after
    each complete TTS utterance.
    """

    def __init__(self, tts_service: MegakernelTTSService):
        self._tts = tts_service
        self._utterance_start: Optional[float] = None  # when user stopped talking
        self._tts_started:     Optional[float] = None
        self._first_audio_out: Optional[float] = None
        self._audio_bytes_total: int = 0
        self._run_count: int = 0

        # Rolling history for summary
        self._history: list[dict] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        now   = time.perf_counter()
        frame = data.frame

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._utterance_start   = now
            self._tts_started       = None
            self._first_audio_out   = None
            self._audio_bytes_total = 0

        elif isinstance(frame, TTSStartedFrame):
            self._tts_started = now

        elif isinstance(frame, TTSAudioRawFrame):
            if self._first_audio_out is None:
                self._first_audio_out = now
            self._audio_bytes_total += len(frame.audio)

        elif isinstance(frame, TTSStoppedFrame):
            self._emit_metrics(now)

    def _emit_metrics(self, now: float) -> None:
        self._run_count += 1

        ttfc_ms = (
            (self._first_audio_out - (self._tts_started or self._first_audio_out)) * 1000
            if self._first_audio_out and self._tts_started
            else self._tts.last_ttfc_ms
        )

        e2e_ms = (
            (self._first_audio_out - self._utterance_start) * 1000
            if self._first_audio_out and self._utterance_start
            else None
        )

        tok_per_s = self._tts.last_tok_per_s
        rtf       = self._tts.last_rtf

        # Audio duration from byte count
        sample_rate  = self._tts.sample_rate
        n_samples    = self._audio_bytes_total // 2   # int16
        audio_dur_s  = n_samples / sample_rate if sample_rate else 0

        record = {
            "run":        self._run_count,
            "ttfc_ms":    ttfc_ms,
            "e2e_ms":     e2e_ms,
            "tok_per_s":  tok_per_s,
            "rtf":        rtf,
            "audio_s":    audio_dur_s,
        }
        self._history.append(record)

        # Pretty-print
        e2e_str = f"{e2e_ms:.0f}ms" if e2e_ms is not None else "N/A"
        print(
            f"\n{'─'*55}\n"
            f"  Utterance #{self._run_count} metrics\n"
            f"  TTFC          : {ttfc_ms:.0f} ms  (target <60ms)\n"
            f"  E2E latency   : {e2e_str}\n"
            f"  RTF           : {rtf:.3f}  (target <0.15)\n"
            f"  Talker tok/s  : {tok_per_s:.0f}  (target ~1000)\n"
            f"  Audio output  : {audio_dur_s:.2f}s\n"
            f"{'─'*55}\n"
        )

    def print_summary(self) -> None:
        """Print aggregate stats over all utterances."""
        if not self._history:
            print("No utterances recorded.")
            return

        ttfc_vals   = [r["ttfc_ms"]   for r in self._history]
        rtf_vals    = [r["rtf"]       for r in self._history if r["rtf"] > 0]
        tok_vals    = [r["tok_per_s"] for r in self._history if r["tok_per_s"] > 0]
        e2e_vals    = [r["e2e_ms"]    for r in self._history if r["e2e_ms"] is not None]

        def _avg(lst): return sum(lst) / len(lst) if lst else 0
        def _min(lst): return min(lst) if lst else 0

        print(
            f"\n{'═'*55}\n"
            f"  AGGREGATE METRICS  ({len(self._history)} utterances)\n"
            f"{'─'*55}\n"
            f"  TTFC  avg={_avg(ttfc_vals):.0f}ms  min={_min(ttfc_vals):.0f}ms\n"
            f"  E2E   avg={_avg(e2e_vals):.0f}ms  min={_min(e2e_vals):.0f}ms\n"
            f"  RTF   avg={_avg(rtf_vals):.3f}  min={_min(rtf_vals):.3f}\n"
            f"  tok/s avg={_avg(tok_vals):.0f}\n"
            f"{'═'*55}\n"
        )
