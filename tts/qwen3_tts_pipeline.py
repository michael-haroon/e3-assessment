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
from typing import AsyncIterator

import numpy as np
import torch
from loguru import logger

TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS    = 1

# ~150 ms per yielded chunk
_CHUNK_SAMPLES = int(TTS_SAMPLE_RATE * 0.15)   # 3600 samples
_CHUNK_BYTES   = _CHUNK_SAMPLES * 2             # int16 -> 7200 bytes

_ADAPTIVE_MAX_NEW_TOKENS = os.getenv("TTS_ADAPTIVE_MAX_NEW_TOKENS", "0") != "0"
_MIN_NEW_TOKENS = int(os.getenv("TTS_MIN_NEW_TOKENS", "96"))
_TOKENS_PER_CHAR = float(os.getenv("TTS_TOKENS_PER_CHAR", "2.6"))
_SENTENCE_BONUS = int(os.getenv("TTS_SENTENCE_TOKEN_BONUS", "24"))
_SHORT_TEXT_CHARS = int(os.getenv("TTS_SHORT_TEXT_CHARS", "80"))
_SHORT_TEXT_MAX_NEW_TOKENS = int(os.getenv("TTS_SHORT_TEXT_MAX_NEW_TOKENS", "180"))

class Qwen3TTSPipeline:
    """
    Qwen3-TTS-12Hz-0.6B-CustomVoice with megakernel LM backend.
    Uses official qwen-tts package: from qwen_tts import Qwen3TTSModel
    Instantiate once; call synthesize_streaming() repeatedly.
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
        self._cuda_poisoned = False

        if verbose:
            logger.info("Qwen3TTSPipeline created -- weights load on first call")

    # -------------------------------------------------------------------------
    # Lazy loading
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
            attn_implementation="sdpa",   # cuDNN FA3 on Blackwell; no flash-attn pkg needed
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

    # -------------------------------------------------------------------------
    # Synthesis
    # -------------------------------------------------------------------------

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Async generator: text -> int16 PCM bytes chunks (24 kHz, mono).
        Runs blocking synthesis in executor, then chunks and yields.
        """
        if self._cuda_poisoned:
            raise RuntimeError("TTS CUDA context is in a failed state; restart the worker process")

        self._ensure_loaded()

        loop = asyncio.get_event_loop()
        target_tokens = self._estimate_max_new_tokens(text)
        logger.debug(f"TTS token budget: {target_tokens} (text_len={len(text)})")
        audio_np, elapsed = await loop.run_in_executor(
            None, self._synthesize_blocking, text, target_tokens
        )

        audio_dur = len(audio_np) / TTS_SAMPLE_RATE
        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0
        logger.info(
            f"TTS: {len(text)} chars -> {audio_dur:.2f}s | "
            f"wall={elapsed*1000:.0f}ms RTF={rtf:.3f}"
        )

        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        for offset in range(0, len(audio_bytes), _CHUNK_BYTES):
            yield audio_bytes[offset : offset + _CHUNK_BYTES]
            await asyncio.sleep(0)

    @staticmethod
    def _is_fatal_cuda_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "cuda" in msg and (
            "illegal memory access" in msg
            or "device-side assert" in msg
            or "device kernel image is invalid" in msg
        )

    def _handle_fatal_cuda_error(self, exc: Exception) -> None:
        self._cuda_poisoned = True
        logger.error(
            "Fatal CUDA error in TTS decode; skipping fallback because CUDA context is likely poisoned. "
            "Please restart the worker process."
        )

    def _synthesize_blocking(self, text: str, max_new_tokens: int) -> tuple:
        """Blocking synthesis -- runs in thread-pool executor."""
        t0 = time.perf_counter()
        try:
            audio = self._synthesize_with_megakernel(text, max_new_tokens)
        except Exception as e:
            if self._is_fatal_cuda_error(e):
                self._handle_fatal_cuda_error(e)
                raise
            logger.warning(f"Megakernel failed ({e!r}) -- falling back to native generate")
            audio = self._synthesize_fallback(text, max_new_tokens)
        return audio, time.perf_counter() - t0


    def _estimate_max_new_tokens(self, text: str) -> int:
        """Estimate per-request token budget tuned for lower conversational latency."""
        if not _ADAPTIVE_MAX_NEW_TOKENS:
            return self.max_new_tokens

        stripped = text.strip()
        n_chars = len(stripped)
        est = int(max(_MIN_NEW_TOKENS, n_chars * _TOKENS_PER_CHAR))
        if stripped.endswith((".", "!", "?")):
            est += _SENTENCE_BONUS

        if n_chars <= _SHORT_TEXT_CHARS:
            est = min(est, _SHORT_TEXT_MAX_NEW_TOKENS)

        return max(1, min(est, self.max_new_tokens))

    def _synthesize_with_megakernel(self, text: str, max_new_tokens: int) -> np.ndarray:
        """
        generate_custom_voice() with megakernel replacing the inner LM decode.

        Qwen3TTSModel stores its backbone transformer at .model.
        We temporarily replace model.model.generate() with our megakernel
        implementation, call generate_custom_voice(), then restore.
        """
        talker   = self._talker
        inner_lm = self._qwen_model.model.talker   # HF backbone (Qwen3 transformer)

        original_generate = inner_lm.generate
        mk_stats: dict = {}

        def _megakernel_generate(inputs_embeds=None, input_ids=None, **kwargs):
            from transformers.generation import GenerateDecoderOnlyOutput

            max_new_tokens = kwargs.get("max_new_tokens", talker.max_new_tokens)
            eos_ids = kwargs.get("eos_token_id", [])
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_set = set(eos_ids)

            # Step 1: PyTorch prefill — run ONE forward pass to get first codec token
            with torch.no_grad():
                out = original_generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    max_new_tokens=1,   # ← only generate ONE token via PyTorch
                    min_new_tokens=1,
                    **{k: v for k, v in kwargs.items() 
                    if k not in ("max_new_tokens", "min_new_tokens", "eos_token_id")}
                )

            # out is shape (1, seq_len+1) — last token is first generated codec token
            first_token = int(out.sequences[0, -1].item())

            if first_token in eos_set:
                return out

            # Step 2: megakernel handles all remaining tokens
            device = inputs_embeds.device
            talker.reset()
            generated = [first_token]
            fake_hidden_list = []
            t0 = time.perf_counter()

            token_id = first_token
            for _ in range(min(max_new_tokens, 600) - 1):
                tok = talker.step(token_id)

                # Capture real hidden state from megakernel buffer
                layer_hidden = talker._hidden.float().unsqueeze(0).unsqueeze(0).clone()  # (1, 1, 1024)

                # codec_ids: first codebook is our token, rest zeros
                codec_ids = torch.zeros(1, 16, dtype=torch.long, device=device)
                codec_ids[0, 0] = tok

                fake_hidden_list.append(((layer_hidden,), codec_ids))
                generated.append(tok)
                token_id = tok
                if tok in eos_set:
                    break

            elapsed = time.perf_counter() - t0
            mk_stats.update(
                tokens=len(generated),
                elapsed=elapsed,
                tok_per_s=len(generated) / elapsed if elapsed > 0 else 0,
            )
            logger.info(f"Megakernel: {len(generated)} tok in {elapsed*1000:.1f}ms "
                        f"({mk_stats['tok_per_s']:.0f} tok/s)")

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
                max_new_tokens=max_new_tokens,
            )
        finally:
            inner_lm.generate = original_generate

        self.last_mk_stats = mk_stats
        audio = wavs[0]
        return audio.squeeze().astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)

    def _synthesize_fallback(self, text: str, max_new_tokens: int) -> np.ndarray:
        """Plain generate_custom_voice() without megakernel (fallback)."""
        wavs, sr = self._qwen_model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            max_new_tokens=max_new_tokens,
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
