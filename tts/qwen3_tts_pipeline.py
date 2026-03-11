"""
qwen3_tts_pipeline.py
─────────────────────
End-to-end Qwen3-TTS inference with true streaming audio output.

Architecture
────────────
Text
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  PyTorch Prefill  (HF generate, max_new_tokens=1)               │
│  Runs the full input-embed sequence (speaker + text tokens)     │
│  through the 20-layer talker to produce the first codec token.  │
│  This is the dominant TTFC contributor (~200–500ms warm).       │
└─────────────────────────────────────────────────────────────────┘
  │  first codec token id
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Megakernel Decode  (TTSTalkerDecoder, ~1ms/token)              │
│  Generates codec tokens in chunks of CHUNK_TOKENS (default 24). │
│  After each chunk the vocoder is called immediately.            │
└─────────────────────────────────────────────────────────────────┘
  │  codec token chunks → vocoder → PCM chunks
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  CosyVoice Vocoder  (speech_tokenizer.decode)                   │
│  Called on cumulative token prefix; only the *new* audio delta  │
│  is yielded per chunk to avoid re-playing earlier audio.        │
└─────────────────────────────────────────────────────────────────┘

Streaming contract
──────────────────
`synthesize_streaming()` is an async generator that yields bytes objects
(int16 PCM, 24 kHz, mono).

True streaming is implemented via asyncio.Queue:
  - A background thread runs the blocking synthesis and puts audio
    chunks into the queue as soon as each CHUNK_TOKENS batch is decoded
    and vocoded.
  - The async generator pulls from the queue and yields immediately.
  - TTFC is therefore: prefill_time + megakernel(CHUNK_TOKENS) + vocoder_first_chunk
    rather than total synthesis time.

TTFC realistic targets (warm model, RTX 5090):
  - Prefill:            ~200–500ms  (HF forward over input embeds, hard floor)
  - Megakernel 24 tok:  ~24ms
  - Vocoder first chunk: ~50–100ms
  - Total TTFC:         ~300–600ms  (warm)  vs ~3500ms with old buffered approach

Notes
─────
- The vocoder (CosyVoice flow-matching) conditions on the cumulative token
  sequence so we call it on the full prefix each time and subtract previously
  yielded samples.  This avoids boundary artifacts at chunk boundaries.
- CHUNK_TOKENS is tunable: smaller = lower TTFC but more vocoder calls.
  Default 24 tokens = 2 s of audio at 12 Hz.
- Model is pre-warmed on __init__ (verbose=True path) to avoid loading
  overhead counting towards the first-call TTFC.
"""

import asyncio
import concurrent.futures
import os
import time
from typing import AsyncIterator

import numpy as np
import torch
from loguru import logger

TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS    = 1

# Tokens per vocoder chunk.  12 tok = 1 s of audio at 12 Hz.
# 24 tok = 2 s — good balance of TTFC vs vocoder-call overhead.
# Lower this to reduce TTFC at cost of more frequent vocoder calls.
_CHUNK_TOKENS = int(os.getenv("TTS_CHUNK_TOKENS", "24"))

# Sentinel placed in the queue to signal end of stream
_DONE = object()

# Dedicated thread pool — one worker is enough since synthesis is single-GPU
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_synth")


class Qwen3TTSPipeline:
    """
    Qwen3-TTS-12Hz-0.6B-CustomVoice with megakernel LM backend and true streaming.

    Instantiate once (weights load on first call or via warm()).
    Call synthesize_streaming() repeatedly.
    """

    MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        speaker: str = "Ryan",      # English voices: Ryan, Aiden
        language: str = "English",
        max_new_tokens: int = int(os.getenv("TTS_MAX_TOKENS", "1500")),
        verbose: bool = True,
    ):
        self.model_name     = model_name
        self.speaker        = speaker
        self.language       = language
        self.max_new_tokens = max_new_tokens
        self.last_mk_stats: dict = {}

        self._qwen_model = None   # Qwen3TTSModel
        self._talker     = None   # TTSTalkerDecoder (megakernel)
        self._loaded     = False

        if verbose:
            logger.info("Qwen3TTSPipeline created -- call warm() or wait for first synthesize_streaming()")

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        logger.info(f"Loading {self.model_name} ...")
        t0 = time.perf_counter()

        from qwen_tts import Qwen3TTSModel
        self._qwen_model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        logger.info(f"Qwen3TTSModel loaded in {time.perf_counter() - t0:.1f}s")

        logger.info("Loading megakernel talker decoder ...")
        t1 = time.perf_counter()
        from megakernel.tts_talker_decoder import TTSTalkerDecoder
        self._talker = TTSTalkerDecoder(
            qwen_model=self._qwen_model,
            verbose=False,
        )
        self._talker.max_new_tokens = self.max_new_tokens
        logger.info(f"Talker decoder loaded in {time.perf_counter() - t1:.1f}s")

        self._loaded = True

    def warm(self) -> None:
        """
        Eagerly load weights and run a short warm-up synthesis so the first
        real call does not pay model-load overhead in its TTFC measurement.
        Call this once at server startup.
        """
        self._ensure_loaded()
        logger.info("Warming up TTS pipeline (short forward pass, 24 tokens max)...")
        t0 = time.perf_counter()
        try:
            # Save and override max_new_tokens so warm-up exits after one chunk
            saved = self.max_new_tokens
            self.max_new_tokens = _CHUNK_TOKENS + 2   # one chunk + a little buffer
            self._synthesize_chunked(
                "Hi.",
                chunk_callback=lambda *a, **kw: None,
            )
        except Exception as e:
            logger.warning(f"Warm-up synthesis failed (non-fatal): {e!r}")
        finally:
            self.max_new_tokens = saved
            if self._talker is not None:
                self._talker.max_new_tokens = saved
        logger.info(f"Warm-up done in {(time.perf_counter() - t0)*1000:.0f}ms")

    # -------------------------------------------------------------------------
    # Public streaming API
    # -------------------------------------------------------------------------

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Async generator: text -> int16 PCM bytes chunks (24 kHz, mono).

        Yields audio as soon as each CHUNK_TOKENS batch is decoded and vocoded.
        TTFC = prefill + megakernel(CHUNK_TOKENS) + first vocoder call.
        """
        self._ensure_loaded()

        loop   = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _callback(audio_bytes: bytes, is_final: bool) -> None:
            """Called from synthesis thread; put chunk into async queue."""
            if audio_bytes:  # skip empty sentinel bytes
                loop.call_soon_threadsafe(queue.put_nowait, audio_bytes)
            if is_final:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        t0 = time.perf_counter()
        first_yielded = False

        # Launch blocking synthesis in thread pool
        future = loop.run_in_executor(
            _EXECUTOR,
            self._synthesize_chunked,
            text,
            _callback,
        )

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if not first_yielded:
                    ttfc_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"TTFC: {ttfc_ms:.0f}ms")
                    first_yielded = True
                yield item
        finally:
            # Ensure synthesis thread completes and surfaces any exceptions
            await future

    # -------------------------------------------------------------------------
    # Blocking synthesis (runs in thread pool)
    # -------------------------------------------------------------------------

    def _synthesize_chunked(self, text: str, chunk_callback) -> None:
        """
        Blocking synthesis with chunked streaming via chunk_callback.

        chunk_callback(audio_bytes: bytes, is_final: bool) is called:
          - Once per CHUNK_TOKENS batch as audio becomes available
          - Once more with is_final=True after the last chunk

        Falls back to full-buffer synthesis if megakernel path fails.
        """
        t0 = time.perf_counter()
        try:
            self._synthesize_with_megakernel_chunked(text, chunk_callback)
        except Exception as e:
            logger.warning(f"Megakernel chunked path failed ({e!r}) -- falling back")
            import traceback; traceback.print_exc()
            audio = self._synthesize_fallback(text)
            audio_bytes = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            chunk_callback(audio_bytes, True)

        elapsed = time.perf_counter() - t0
        logger.info(f"_synthesize_chunked wall={elapsed*1000:.0f}ms")

    def _synthesize_with_megakernel_chunked(self, text: str, chunk_callback) -> None:
        """
        Core streaming synthesis:
          1. PyTorch prefill (HF generate, max_new_tokens=1) → first codec token
          2. Megakernel decode in batches of CHUNK_TOKENS
          3. After each batch: call vocoder on cumulative tokens, yield new audio delta

        Audio is streamed via chunk_callback as it is produced.
        The outer generate_custom_voice() vocoder call is intercepted and skipped
        because we have already vocoded and streamed all audio internally.
        """
        talker           = self._talker
        inner_lm         = self._qwen_model.model.talker
        speech_tokenizer = self._qwen_model.model.speech_tokenizer

        original_generate        = inner_lm.generate
        original_decode          = speech_tokenizer.decode
        mk_stats: dict = {}

        # ------------------------------------------------------------------
        # Patch speech_tokenizer.decode to be a no-op after streaming is done.
        # The outer generate_custom_voice() calls decode() on the full token
        # sequence AFTER _streaming_generate returns — we've already vocoded
        # everything incrementally, so skip that redundant full-sequence call.
        # We return a minimal valid response: one zero-sample waveform.
        # ------------------------------------------------------------------
        _decode_suppressed = [False]

        def _noop_decode(codes_list, *args, **kwargs):
            if _decode_suppressed[0]:
                # Return zero-length audio so the outer code doesn't crash
                dummy = np.zeros(0, dtype=np.float32)
                return [dummy] * len(codes_list), TTS_SAMPLE_RATE
            return original_decode(codes_list, *args, **kwargs)

        # ------------------------------------------------------------------
        # Patch inner_lm.generate to intercept the inputs_embeds + run
        # prefill for 1 token only, then hand off to megakernel
        # ------------------------------------------------------------------
        def _streaming_generate(inputs_embeds=None, input_ids=None, **kwargs):
            from transformers.generation import GenerateDecoderOnlyOutput

            max_new_tokens = kwargs.get("max_new_tokens", talker.max_new_tokens)
            eos_ids = kwargs.get("eos_token_id", [])
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_set = set(eos_ids)

            # ── Step 1: PyTorch prefill for 1 token ──────────────────────
            t_prefill = time.perf_counter()
            with torch.no_grad():
                out = original_generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("max_new_tokens", "min_new_tokens", "eos_token_id")}
                )
            prefill_ms = (time.perf_counter() - t_prefill) * 1000
            logger.info(f"Prefill: {prefill_ms:.0f}ms")

            first_token = int(out.sequences[0, -1].item())

            if first_token in eos_set:
                return out

            # ── Step 2: megakernel decode in CHUNK_TOKENS batches ────────
            device = inputs_embeds.device
            talker.reset()
            generated   = [first_token]
            fake_hidden_list = []
            samples_yielded  = 0   # samples already sent to caller

            t_mk = time.perf_counter()
            token_id = first_token

            cap = min(max_new_tokens, 1500) - 1

            for step_i in range(cap):
                tok = talker.step(token_id)

                layer_hidden = talker._hidden.float().unsqueeze(0).unsqueeze(0).clone()
                codec_ids    = torch.zeros(1, 16, dtype=torch.long, device=device)
                codec_ids[0, 0] = tok
                fake_hidden_list.append(((layer_hidden,), codec_ids))
                generated.append(tok)
                token_id = tok

                is_eos   = (tok in eos_set)
                is_chunk = ((step_i + 1) % _CHUNK_TOKENS == 0)

                if is_chunk or is_eos:
                    # ── vocoder: decode cumulative token prefix ───────────
                    codes_tensor = torch.tensor(
                        generated, dtype=torch.long, device=device
                    ).unsqueeze(-1).expand(-1, 16).unsqueeze(0)
                    # shape: (1, T, 16) — first codebook is real, rest zeros
                    codes_full = torch.zeros(
                        1, len(generated), 16, dtype=torch.long, device=device
                    )
                    codes_full[0, :, 0] = torch.tensor(
                        generated, dtype=torch.long, device=device
                    )

                    try:
                        wavs, _sr = speech_tokenizer.decode(
                            [{"audio_codes": codes_full[0]}]
                        )
                        audio_np = wavs[0]
                        if audio_np.ndim > 1:
                            audio_np = audio_np.squeeze()
                        audio_np = audio_np.astype(np.float32)

                        # Emit only the NEW samples since last chunk
                        new_audio = audio_np[samples_yielded:]
                        samples_yielded = len(audio_np)

                        if len(new_audio) > 0:
                            audio_bytes = (
                                (new_audio * 32767)
                                .clip(-32768, 32767)
                                .astype(np.int16)
                                .tobytes()
                            )
                            chunk_callback(audio_bytes, False)
                    except Exception as voc_err:
                        logger.warning(f"Vocoder chunk failed: {voc_err!r}")

                if is_eos:
                    break

            elapsed_mk = time.perf_counter() - t_mk
            mk_stats.update(
                tokens=len(generated),
                elapsed=elapsed_mk,
                tok_per_s=len(generated) / elapsed_mk if elapsed_mk > 0 else 0,
            )
            logger.info(
                f"Megakernel: {len(generated)} tok in {elapsed_mk*1000:.1f}ms "
                f"({mk_stats['tok_per_s']:.0f} tok/s)"
            )

            # Signal end-of-stream (empty bytes, is_final=True)
            chunk_callback(b"", True)

            # Suppress the outer vocoder call that generate_custom_voice will
            # make on the hidden_states we return — audio already streamed.
            _decode_suppressed[0] = True

            return GenerateDecoderOnlyOutput(
                sequences=torch.tensor([generated], dtype=torch.long, device=device),
                hidden_states=tuple(fake_hidden_list),
            )

        inner_lm.generate         = _streaming_generate
        speech_tokenizer.decode  = _noop_decode
        try:
            # generate_custom_voice internally calls inner_lm.generate
            # (now _streaming_generate) which streams all audio via chunk_callback,
            # then calls speech_tokenizer.decode (now _noop_decode) which is skipped.
            self._qwen_model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                max_new_tokens=self.max_new_tokens,
            )
        finally:
            inner_lm.generate        = original_generate
            speech_tokenizer.decode  = original_decode

        self.last_mk_stats = mk_stats

    # -------------------------------------------------------------------------
    # Fallback (no megakernel)
    # -------------------------------------------------------------------------

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """Plain generate_custom_voice() without megakernel."""
        wavs, sr = self._qwen_model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
        )
        self.last_mk_stats = {"fallback": True}
        audio = wavs[0]
        return audio.squeeze().astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return TTS_SAMPLE_RATE

    @property
    def channels(self) -> int:
        return TTS_CHANNELS
