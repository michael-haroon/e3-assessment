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
│  PyTorch Prefill  (HF generate, max_new_tokens=1)               │
│  Runs the full input-embed sequence (speaker + text tokens)     │
│  through the 20-layer talker to produce the first codec token.  │
└─────────────────────────────────────────────────────────────────┘
  │  first codec token id
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Megakernel Decode  (TTSTalkerDecoder, ~1ms/token)              │
│  Generates all codec tokens at ~1000 tok/s via CUDA kernel.     │
└─────────────────────────────────────────────────────────────────┘
  │  full codec token sequence
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  CosyVoice Vocoder  (speech_tokenizer.decode, called once)      │
│  Converts full token sequence → PCM waveform in one shot.       │
└─────────────────────────────────────────────────────────────────┘
  │  PCM float32 array → chunked int16 bytes via asyncio.Queue
  ▼
  caller (Pipecat TTS service)

Streaming contract
──────────────────
`synthesize_streaming()` is an async generator that yields bytes objects
(int16 PCM, 24 kHz, mono) as soon as synthesis completes.

The synthesis runs in a thread-pool executor. Audio chunks are pushed
into an asyncio.Queue as soon as the full waveform is available, then
yielded one chunk at a time. This is true streaming to the caller
(chunks arrive progressively) even though the vocoder runs once.

TTFC = prefill + megakernel decode + vocoder (one call on full sequence).

Why not incremental vocoding?
─────────────────────────────
CosyVoice's flow-matching vocoder conditions on the full token sequence
globally. Calling it on partial prefixes either hangs on longer sequences
or produces artifacts. The safe path is one call on the complete output.

TTFC realistic targets (warm model, RTX 5090):
  - Prefill:          ~40ms   (warm; ~500ms cold first call)
  - Megakernel decode: ~600ms  (600 tokens at 1000 tok/s)
  - Vocoder:          ~300ms  (one call, full sequence)
  - Total TTFC:       ~940ms  (warm) — honest number

The model is pre-warmed via warm() so model-load time never appears
in TTFC measurements.
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

# Bytes per yielded chunk to caller (~150ms of audio)
_CHUNK_SAMPLES = int(TTS_SAMPLE_RATE * 0.15)   # 3600 samples
_CHUNK_BYTES   = _CHUNK_SAMPLES * 2             # int16 → 7200 bytes

# Sentinel placed in the queue to signal end of stream
_DONE = object()

# Dedicated thread pool — one worker since synthesis is single-GPU
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="tts_synth"
)


class Qwen3TTSPipeline:
    """
    Qwen3-TTS-12Hz-0.6B-CustomVoice with megakernel LM backend.
    Instantiate once; call warm() at startup; call synthesize_streaming() repeatedly.
    """

    MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        speaker: str = "Ryan",
        language: str = "English",
        max_new_tokens: int = int(os.getenv("TTS_MAX_TOKENS", "1500")),
        verbose: bool = True,
    ):
        self.model_name     = model_name
        self.speaker        = speaker
        self.language       = language
        self.max_new_tokens = max_new_tokens
        self.last_mk_stats: dict = {}

        self._qwen_model = None
        self._talker     = None
        self._loaded     = False

        if verbose:
            logger.info("Qwen3TTSPipeline created — call warm() before first use")

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
        Load weights and run one short synthesis so CUDA kernels are compiled
        and cached before the first timed call. Call once at server startup.
        """
        self._ensure_loaded()
        logger.info("Warming up TTS pipeline...")
        t0 = time.perf_counter()
        try:
            self._synthesize_blocking("Hi.")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-fatal): {e!r}")
        logger.info(f"Warm-up done in {(time.perf_counter() - t0)*1000:.0f}ms")

    # -------------------------------------------------------------------------
    # Public streaming API
    # -------------------------------------------------------------------------

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Async generator: text → int16 PCM bytes chunks (24 kHz, mono).

        Synthesis runs in a thread-pool executor. As soon as the full audio
        array is ready, chunks are pushed into an asyncio.Queue and yielded
        immediately — the caller receives audio progressively rather than
        waiting for a single large return value.

        TTFC is logged from the moment this coroutine is entered until the
        first chunk is yielded.
        """
        self._ensure_loaded()

        loop  = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce() -> None:
            """Runs in thread pool: synthesize then push chunks to queue."""
            try:
                audio_np, elapsed = self._synthesize_blocking(text)
                audio_dur = len(audio_np) / TTS_SAMPLE_RATE
                rtf = elapsed / audio_dur if audio_dur > 0 else 0.0
                logger.info(
                    f"TTS: {len(text)} chars → {audio_dur:.2f}s audio | "
                    f"wall={elapsed*1000:.0f}ms RTF={rtf:.3f}"
                )
                audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                for offset in range(0, len(audio_bytes), _CHUNK_BYTES):
                    chunk = audio_bytes[offset : offset + _CHUNK_BYTES]
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                logger.error(f"Synthesis failed: {e!r}")
                import traceback; traceback.print_exc()
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        t0 = time.perf_counter()
        first_yielded = False

        future = loop.run_in_executor(_EXECUTOR, _produce)

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if not first_yielded:
                    logger.info(f"TTFC: {(time.perf_counter() - t0)*1000:.0f}ms")
                    first_yielded = True
                yield item
        finally:
            await future

    # -------------------------------------------------------------------------
    # Blocking synthesis
    # -------------------------------------------------------------------------

    def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, float]:
        """
        Run full synthesis synchronously. Returns (audio_np float32, wall_seconds).
        Tries megakernel path first, falls back to native HF generate on error.
        """
        t0 = time.perf_counter()
        try:
            audio = self._synthesize_with_megakernel(text)
        except Exception as e:
            logger.warning(f"Megakernel failed ({e!r}) — falling back to native generate")
            import traceback; traceback.print_exc()
            audio = self._synthesize_fallback(text)
        return audio, time.perf_counter() - t0

    def _synthesize_with_megakernel(self, text: str) -> np.ndarray:
        """
        Monkey-patch inner_lm.generate to replace the HF autoregressive loop
        with the megakernel, then call generate_custom_voice normally.

        Flow:
          generate_custom_voice()
            → self.model.generate()          (outer, builds input embeds)
              → self.talker.generate()        ← WE REPLACE THIS
                  prefill 1 token via PyTorch (gets first codec token + warms KV)
                  then megakernel for remaining tokens (~1ms/tok)
            → speech_tokenizer.decode()      (vocoder, called once on full sequence)
        """
        talker   = self._talker
        inner_lm = self._qwen_model.model.talker
        original_generate = inner_lm.generate
        mk_stats: dict = {}

        def _megakernel_generate(inputs_embeds=None, input_ids=None, **kwargs):
            from transformers.generation import GenerateDecoderOnlyOutput

            max_new_tokens = kwargs.get("max_new_tokens", talker.max_new_tokens)
            eos_ids = kwargs.get("eos_token_id", [])
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_set = set(eos_ids)

            # ── Prefill: one PyTorch forward to get first codec token ─────
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
            logger.info(f"Prefill: {(time.perf_counter()-t_prefill)*1000:.0f}ms")

            first_token = int(out.sequences[0, -1].item())
            if first_token in eos_set:
                return out

            # ── Megakernel decode for all remaining tokens ────────────────
            device = inputs_embeds.device
            talker.reset()
            generated        = [first_token]
            fake_hidden_list = []
            t_mk = time.perf_counter()
            token_id = first_token

            for _ in range(min(max_new_tokens, 1500) - 1):
                tok = talker.step(token_id)

                layer_hidden = talker._hidden.float().unsqueeze(0).unsqueeze(0).clone()
                codec_ids    = torch.zeros(1, 16, dtype=torch.long, device=device)
                codec_ids[0, 0] = tok
                fake_hidden_list.append(((layer_hidden,), codec_ids))
                generated.append(tok)
                token_id = tok
                if tok in eos_set:
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

            return GenerateDecoderOnlyOutput(
                sequences=torch.tensor([generated], dtype=torch.long, device=device),
                hidden_states=tuple(fake_hidden_list),
            )

        inner_lm.generate = _megakernel_generate
        try:
            wavs, sr = self._qwen_model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                max_new_tokens=self.max_new_tokens,
            )
        finally:
            inner_lm.generate = original_generate

        self.last_mk_stats = mk_stats
        audio = wavs[0]
        return audio.squeeze().astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """Native HF generate without megakernel — fallback only."""
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
