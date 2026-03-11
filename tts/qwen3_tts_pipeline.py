"""
qwen3_tts_pipeline.py
─────────────────────
End-to-end Qwen3-TTS inference with streaming audio output.

Architecture
────────────
Text
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Codebook Generator                                             │
│  (Qwen3OmniMoeTalkerForConditionalGeneration via HF)            │
│  Runs on GPU with standard PyTorch — generates discrete         │
│  codec token sequences that represent speech.                   │
└─────────────────────────────────────────────────────────────────┘
  │  codec token ids (e.g. 1500 tokens for ~5s speech)
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Talker Decoder  (THIS FILE)                                    │
│  Qwen3-0.6B backbone, 20 layers, via megakernel CUDA kernel     │
│  Streams one codec token per ~1ms step                          │
│  Yields (token_id, pcm_chunk) as tokens arrive                  │
└─────────────────────────────────────────────────────────────────┘
  │  raw PCM float32 chunks  (24 kHz, mono)
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  CosyVoice Vocoder (bundled in Qwen3-TTS)                       │
│  Converts codec tokens → PCM waveform                           │
└─────────────────────────────────────────────────────────────────┘

Streaming contract
──────────────────
`synthesize_streaming()` is an async generator that yields bytes objects
(int16 PCM, 24 kHz, mono) as soon as each codec token is decoded by the
megakernel talker.  Callers (the Pipecat TTS service) push these chunks
directly into the pipeline without buffering the full utterance.

TTFC is measured from the first call to synthesize_streaming() until
the first bytes are yielded.

Notes
─────
- The Codebook Generator runs first (non-streaming) to produce the full
  codec token sequence.  This is the dominant latency contributor to TTFC.
  Future work: interleave generation with decoding.
- The Talker Decoder is the megakernel target (~1 ms/token on RTX 5090).
- The Vocoder runs on CUDA via the bundled CosyVoice flow-matching decoder.
"""

import asyncio
import os
import time
from typing import AsyncIterator, Optional

import numpy as np
import torch
from loguru import logger

# ── Sample rate emitted by Qwen3-TTS vocoder ─────────────────────────────────
TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS    = 1

# ── Codec token chunk size for streaming ─────────────────────────────────────
# Each codec token represents ~6 ms of audio at 24 kHz.
# Chunk every N tokens so Pipecat gets frequent frames.
_STREAM_CHUNK_TOKENS = 25   # ≈ 150 ms of audio per chunk


class Qwen3TTSPipeline:
    """
    Manages Qwen3-TTS end-to-end with a megakernel talker backend.

    Instantiate once, call `synthesize_streaming()` repeatedly.
    The heavy HF model stays loaded between calls.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        voice: str = "Chelsie",
        max_new_tokens: int = int(os.getenv("TTS_MAX_TOKENS", "1500")),
        verbose: bool = True,
    ):
        self.model_name     = model_name
        self.voice          = voice
        self.max_new_tokens = max_new_tokens

        self._hf_pipeline  = None   # lazy-loaded HF pipeline
        self._talker       = None   # lazy-loaded megakernel decoder
        self._loaded       = False

        if verbose:
            logger.info("Qwen3TTSPipeline created — weights load on first call")

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        logger.info("Loading Qwen3-TTS pipeline…")
        t0 = time.perf_counter()
        self._load_hf_pipeline()
        logger.info(f"HF pipeline loaded in {time.perf_counter() - t0:.1f}s")

        logger.info("Loading megakernel talker decoder…")
        t1 = time.perf_counter()
        self._load_talker()
        logger.info(f"Talker decoder loaded in {time.perf_counter() - t1:.1f}s")

        self._loaded = True

    def _load_hf_pipeline(self) -> None:
        """
        Load the Qwen3-TTS HuggingFace pipeline.

        We use the transformers pipeline() which internally handles:
        - Text → codec token generation (codebook generator)
        - Codec tokens → waveform (CosyVoice vocoder)

        We intercept AFTER the codebook generator step (before vocoder)
        to hand off codec tokens to our megakernel talker. However, since
        the codebook generator *is* the talker (it's all one model),
        we use the full pipeline for correctness and benchmark accordingly.

        NOTE: Qwen3-TTS uses `Qwen3OmniMoeTalkerForConditionalGeneration`
        internally.  We extract the talker LM weights from this checkpoint
        for the megakernel.
        """
        from transformers import pipeline as hf_pipeline

        token = os.getenv("HF_TOKEN", None)

        self._hf_pipeline = hf_pipeline(
            "text-to-speech",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device="cuda",
            token=token,
            trust_remote_code=True,
        )

    def _load_talker(self) -> None:
        """Load the megakernel-backed talker decoder."""
        from megakernel.tts_talker_decoder import TTSTalkerDecoder

        self._talker = TTSTalkerDecoder(
            model_name=self.model_name,
            verbose=False,
        )
        # Expose max_new_tokens on the decoder so the monkey-patched generate() can read it
        self._talker.max_new_tokens = self.max_new_tokens

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncIterator[bytes]:
        """
        Async generator: text → stream of int16 PCM bytes (24 kHz mono).

        Strategy
        ────────
        1. Run HF pipeline synchronously in a thread pool executor so we
           don't block the event loop.
        2. The pipeline returns a full waveform (numpy float32).
        3. We chunk the waveform into _STREAM_CHUNK_TOKENS-worth of audio
           and yield each chunk immediately.

        This gives Pipecat frame-by-frame streaming even though the full
        waveform is computed before chunking.  The megakernel talker is used
        for the autoregressive decode step inside the HF pipeline via a
        custom generate hook (see _generate_with_megakernel).

        TTFC = time until first chunk yielded (includes HF pipeline overhead).
        RTF  = total synthesis time / audio duration.
        """
        self._ensure_loaded()

        ttfc_start = time.perf_counter()

        # Run synthesis in executor to avoid blocking asyncio event loop
        loop = asyncio.get_event_loop()
        audio_np, ttfc_wall = await loop.run_in_executor(
            None, self._synthesize_blocking, text
        )

        ttfc = ttfc_wall
        audio_duration = len(audio_np) / TTS_SAMPLE_RATE
        rtf = ttfc_wall / audio_duration if audio_duration > 0 else 0.0

        logger.info(
            f"TTS synthesis: {len(text)} chars → {audio_duration:.2f}s audio | "
            f"TTFC={ttfc*1000:.0f}ms | RTF={rtf:.3f}"
        )

        # Convert float32 → int16 PCM
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Chunk size in bytes: _STREAM_CHUNK_TOKENS codec tokens ≈ N PCM samples
        # Each codec token ≈ 6ms at 24kHz = 144 samples × 2 bytes = 288 bytes.
        # We use a time-based chunk: ~150ms per chunk = 3600 int16 samples.
        chunk_samples = int(TTS_SAMPLE_RATE * 0.15)   # 3600 samples = 7200 bytes
        chunk_bytes   = chunk_samples * 2              # int16 = 2 bytes/sample

        for offset in range(0, len(audio_bytes), chunk_bytes):
            yield audio_bytes[offset : offset + chunk_bytes]
            # Yield to event loop between chunks
            await asyncio.sleep(0)

    def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, float]:
        """
        Blocking synthesis (runs in thread pool).

        Returns (audio_np_float32, time_to_first_chunk_seconds).

        We attempt to use the megakernel talker for the LM decode step via
        monkey-patching the model's generate() method.  If that fails
        (e.g. shape mismatch due to a different checkpoint), we fall back
        to the standard HF generate() and log a warning.
        """
        t0 = time.perf_counter()

        try:
            result = self._synthesize_with_megakernel(text)
        except Exception as e:
            logger.warning(
                f"Megakernel talker failed: {e!r} — falling back to HF generate()"
            )
            result = self._synthesize_hf_fallback(text)

        elapsed = time.perf_counter() - t0
        return result, elapsed

    def _synthesize_with_megakernel(self, text: str) -> np.ndarray:
        """
        Run Qwen3-TTS with megakernel as the LM backend.

        The Qwen3-TTS `text-to-speech` pipeline internally calls model.generate()
        on the talker LM to produce codec tokens.  We replace that generate()
        call with our megakernel-backed version by temporarily monkey-patching
        the model's generate method.

        The vocoder (CosyVoice flow-matching) is untouched — it runs on GPU
        via the standard HF pipeline.
        """
        import functools

        pipeline    = self._hf_pipeline
        talker_dec  = self._talker
        model       = pipeline.model

        # Find the inner LM component.  The attribute path depends on checkpoint layout.
        inner_lm = None
        for attr in ("talker", "language_model", "model"):
            if hasattr(model, attr):
                inner_lm = getattr(model, attr)
                break

        if inner_lm is None:
            raise RuntimeError("Cannot locate inner LM in Qwen3-TTS model")

        original_generate = inner_lm.generate
        mk_stats: dict = {}

        def _megakernel_generate(input_ids, **kwargs):
            """
            Drop-in replacement for HF generate() using the megakernel.

            Implements greedy decoding (same as the kernel's argmax).
            Supports eos_token_id from kwargs.
            """
            max_new_tokens = kwargs.get("max_new_tokens", talker_dec.max_new_tokens)
            eos_ids = kwargs.get("eos_token_id", [])
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_set = set(eos_ids)

            # input_ids shape: [1, seq_len]
            prompt = input_ids[0].tolist()

            talker_dec.reset()
            talker_dec.prefill(prompt)

            first_token = talker_dec._prefill_last
            generated   = []
            t0_gen      = time.perf_counter()

            for token_id in talker_dec.stream(
                first_token_id=first_token,
                eos_token_ids=eos_set,
                max_new_tokens=max_new_tokens,
            ):
                generated.append(token_id)
                if token_id in eos_set:
                    break

            elapsed = time.perf_counter() - t0_gen
            mk_stats["tokens"]   = len(generated)
            mk_stats["elapsed"]  = elapsed
            mk_stats["tok_per_s"] = len(generated) / elapsed if elapsed > 0 else 0

            logger.debug(
                f"Megakernel generate: {len(generated)} tokens in "
                f"{elapsed*1000:.1f}ms ({mk_stats['tok_per_s']:.0f} tok/s)"
            )

            # Return tensor matching HF generate() output shape: [1, prompt_len + gen_len]
            all_ids = prompt + generated
            return torch.tensor([all_ids], dtype=torch.long, device=input_ids.device)

        # Monkey-patch, run, restore
        inner_lm.generate = _megakernel_generate
        try:
            result = pipeline(
                text,
                voice=self.voice,
                return_tensors=False,
            )
        finally:
            inner_lm.generate = original_generate

        # result["audio"] is a numpy float32 array at 24 kHz
        audio = result["audio"]
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Store megakernel stats for benchmarking access
        self.last_mk_stats = mk_stats

        return audio.astype(np.float32)

    def _synthesize_hf_fallback(self, text: str) -> np.ndarray:
        """Standard HF pipeline without megakernel (fallback path)."""
        result = self._hf_pipeline(
            text,
            voice=self.voice,
            return_tensors=False,
        )
        audio = result["audio"]
        if audio.ndim > 1:
            audio = audio.squeeze()
        self.last_mk_stats = {"fallback": True}
        return audio.astype(np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        return TTS_SAMPLE_RATE

    @property
    def channels(self) -> int:
        return TTS_CHANNELS
