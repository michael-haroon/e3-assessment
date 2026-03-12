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
  │  codec token ids
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Talker Decoder  (megakernel CUDA kernel)                       │
│  Qwen3-0.6B backbone, 20 layers                                 │
│  Streams one codec token per ~1ms step                          │
└─────────────────────────────────────────────────────────────────┘
  │  raw PCM float32 chunks  (24 kHz, mono)
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  CosyVoice Vocoder (bundled in Qwen3-TTS)                       │
│  Converts codec tokens → PCM waveform                           │
└─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import os
import sys
import time
from typing import AsyncIterator

import numpy as np
import torch
from loguru import logger

# ── flash_attn shim — must run before any qwen_tts import ───────────────────
import bootstrap  # noqa: F401  (side-effect import: registers flash_attn_3 as flash_attn)

TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS    = 1

# ~150 ms per yielded chunk
_CHUNK_SAMPLES = int(TTS_SAMPLE_RATE * 0.15)   # 3600 samples
_CHUNK_BYTES   = _CHUNK_SAMPLES * 2             # int16 -> 7200 bytes

# Reasonable upper bound — prevents runaway generation.
# At ~120 codec tok/s, 300 tokens ≈ 2.5s. Raise if you need longer utterances.
_MAX_TOKENS_HARD_CAP = 300
_DEFAULT_MAX_NEW_TOKENS = 40


class Qwen3TTSPipeline:
    """
    Qwen3-TTS-12Hz-0.6B-CustomVoice with megakernel LM backend.
    Instantiate once; call synthesize_streaming() repeatedly.
    """

    MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        speaker: str = "Ryan",
        language: str = "English",
        max_new_tokens: int = int(os.getenv("TTS_MAX_TOKENS", str(_DEFAULT_MAX_NEW_TOKENS))),
        verbose: bool = True,
    ):
        self.model_name     = model_name
        self.speaker        = speaker
        self.language       = language
        # Clamp to the hard cap so callers can't accidentally pass 1500 again
        self.max_new_tokens = min(max_new_tokens, _MAX_TOKENS_HARD_CAP)
        self.last_mk_stats: dict = {}

        self._qwen_model = None
        self._talker     = None
        self._loaded     = False

        if verbose:
            logger.info(
                f"Qwen3TTSPipeline created — max_new_tokens={self.max_new_tokens} "
                f"(hard cap={_MAX_TOKENS_HARD_CAP})"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy loading
    # ──────────────────────────────────────────────────────────────────────────

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
        # Sync max_new_tokens onto the talker
        self._talker.max_new_tokens = self.max_new_tokens
        logger.info(f"Talker decoder loaded in {time.perf_counter() - t1:.1f}s")

        self._loaded = True

    # ──────────────────────────────────────────────────────────────────────────
    # Synthesis
    # ──────────────────────────────────────────────────────────────────────────

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Async generator: text -> int16 PCM bytes chunks (24 kHz, mono).
        Runs blocking synthesis in executor, then chunks and yields.
        """
        self._ensure_loaded()

        loop = asyncio.get_event_loop()
        audio_np, elapsed = await loop.run_in_executor(
            None, self._synthesize_blocking, text
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

    def _synthesize_blocking(self, text: str) -> tuple:
        """Blocking synthesis — runs in thread-pool executor."""
        t0 = time.perf_counter()
        try:
            audio = self._synthesize_with_megakernel(text)
        except Exception as e:
            logger.warning(f"Megakernel failed ({e!r}) — falling back to native generate")
            audio = self._synthesize_fallback(text)
        return audio, time.perf_counter() - t0

    def _synthesize_with_megakernel(self, text: str) -> np.ndarray:
        """
        generate_custom_voice() with megakernel replacing the inner LM decode loop.
        """
        talker   = self._talker
        inner_lm = self._qwen_model.model.talker

        # Always keep talker in sync with current max_new_tokens
        talker.max_new_tokens = self.max_new_tokens

        original_generate = inner_lm.generate
        mk_stats: dict = {}

        def _copy_prefill_kv_to_talker(prefill_out) -> int:
            """Copy HF prefill cache into megakernel flat KV buffers."""
            pkv = getattr(prefill_out, "past_key_values", None)
            if pkv is None:
                return 0

            # HF can return either a DynamicCache-like object or a tuple/list.
            if hasattr(pkv, "to_legacy_cache"):
                pkv = pkv.to_legacy_cache()
            elif hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                pkv = list(zip(pkv.key_cache, pkv.value_cache))

            if not isinstance(pkv, (list, tuple)):
                return 0

            seq_len = 0
            n_layers = min(len(pkv), talker._k_cache.shape[0])
            for layer_idx in range(n_layers):
                layer = pkv[layer_idx]
                if not isinstance(layer, (list, tuple)) or len(layer) < 2:
                    continue
                k, v = layer[0], layer[1]
                if k is None or v is None:
                    continue

                # Expected HF shape: [B, KV_HEADS, T, HEAD_DIM]
                if k.dim() != 4 or v.dim() != 4:
                    continue
                k = k[:1].to(device=talker._k_cache.device, dtype=talker._k_cache.dtype)
                v = v[:1].to(device=talker._v_cache.device, dtype=talker._v_cache.dtype)

                t = min(k.shape[2], talker._k_cache.shape[2])
                h = min(k.shape[1], talker._k_cache.shape[1])
                d = min(k.shape[3], talker._k_cache.shape[3])
                talker._k_cache[layer_idx, :h, :t, :d].copy_(k[0, :h, :t, :d])
                talker._v_cache[layer_idx, :h, :t, :d].copy_(v[0, :h, :t, :d])
                seq_len = max(seq_len, t)

            talker._position = seq_len
            return seq_len

        def _predict_residual_codebooks(first_codebook_token: int, hidden_1024: torch.Tensor) -> torch.Tensor:
            """Predict codebooks 2..16 using the talker code_predictor path."""
            num_groups = int(getattr(inner_lm.config, "num_code_groups", 16))
            codec_ids = torch.zeros(1, num_groups, dtype=torch.long, device=hidden_1024.device)
            codec_ids[0, 0] = int(first_codebook_token)

            code_predictor = getattr(inner_lm, "code_predictor", None)
            if code_predictor is None:
                return codec_ids

            with torch.no_grad():
                input_ids = torch.tensor([[int(first_codebook_token)]], dtype=torch.long, device=hidden_1024.device)
                last_id_hidden = inner_lm.get_input_embeddings()(input_ids)
                predictor_inputs = torch.cat((hidden_1024.unsqueeze(1), last_id_hidden), dim=1)

                predictor_result = code_predictor.generate(
                    inputs_embeds=predictor_inputs,
                    max_new_tokens=num_groups - 1,
                    do_sample=kwargs.get("subtalker_dosample", False),
                    top_p=kwargs.get("subtalker_top_p", 1.0),
                    top_k=kwargs.get("subtalker_top_k", 0),
                    temperature=kwargs.get("subtalker_temperature", 1.0),
                    return_dict_in_generate=True,
                )

                # `sequences` may include prompt tokens depending on generate impl.
                residual = predictor_result.sequences
                if residual.shape[-1] > (num_groups - 1):
                    residual = residual[..., -(num_groups - 1):]
                residual = residual.to(device=hidden_1024.device, dtype=torch.long)
                codec_ids[0, 1 : 1 + residual.shape[-1]] = residual[0]

            return codec_ids

        def _megakernel_generate(inputs_embeds=None, input_ids=None, **kwargs):
            from transformers.generation import GenerateDecoderOnlyOutput

            # ── Token limit ───────────────────────────────────────────────────
            # BEFORE: min(max_new_tokens, 600) — the 600 hardcap overrode
            # everything and caused 47s audio blobs.
            # NOW: respect whatever was passed, clamped to _MAX_TOKENS_HARD_CAP.
            max_tokens = min(
                kwargs.get("max_new_tokens", talker.max_new_tokens),
                _MAX_TOKENS_HARD_CAP,
            )

            # ── EOS token set ─────────────────────────────────────────────────
            raw_eos = kwargs.get("eos_token_id", [])
            if isinstance(raw_eos, int):
                raw_eos = [raw_eos]
            elif isinstance(raw_eos, torch.Tensor):
                raw_eos = raw_eos.tolist()
            eos_set = set(int(x) for x in raw_eos if x is not None)

            # ── Step 1: PyTorch prefill (one token) ───────────────────────────
            with torch.no_grad():
                out = original_generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("max_new_tokens", "min_new_tokens",
                                    "eos_token_id")}
                )

            first_token = int(out.sequences[0, -1].item())

            # Early exit if we already hit EOS on the first token
            if eos_set and first_token in eos_set:
                logger.debug("Megakernel: EOS on first token — empty utterance")
                return out

            # ── Step 2: megakernel decode loop ────────────────────────────────
            device = inputs_embeds.device
            talker.reset()
            prefill_seq_len = _copy_prefill_kv_to_talker(out)
            if prefill_seq_len > 0:
                logger.debug(f"Megakernel: imported prefill KV cache (seq_len={prefill_seq_len})")
            generated    = [first_token]
            fake_hiddens = []
            t0           = time.perf_counter()
            token_id     = first_token

            # max_tokens - 1 because we already have first_token
            for step in range(max_tokens - 1):
                tok = talker.step(token_id)

                layer_hidden = (
                    talker._hidden.float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .clone()   # (1, 1, 1024)
                )
                codec_ids = _predict_residual_codebooks(
                    first_codebook_token=tok,
                    hidden_1024=talker._hidden,
                )
                fake_hiddens.append(((layer_hidden,), codec_ids))

                generated.append(tok)
                token_id = tok

                # Stop as soon as we see EOS — don't pad to the limit
                if eos_set and tok in eos_set:
                    logger.debug(f"Megakernel: EOS at step {step+1}/{max_tokens}")
                    break

            elapsed = time.perf_counter() - t0
            mk_stats.update(
                tokens    = len(generated),
                elapsed   = elapsed,
                tok_per_s = len(generated) / elapsed if elapsed > 0 else 0,
                hit_eos   = (eos_set and generated[-1] in eos_set),
            )
            logger.info(
                f"Megakernel: {len(generated)} tok in {elapsed*1000:.1f}ms "
                f"({mk_stats['tok_per_s']:.0f} tok/s) "
                f"EOS={'✅' if mk_stats['hit_eos'] else '❌ hit token limit'}"
            )

            return GenerateDecoderOnlyOutput(
                sequences    = torch.tensor(
                    [generated], dtype=torch.long, device=device
                ),
                hidden_states = tuple(fake_hiddens),
            )

        inner_lm.generate = _megakernel_generate
        try:
            wavs, sr = self._qwen_model.generate_custom_voice(
                text           = text,
                language       = self.language,
                speaker        = self.speaker,
                max_new_tokens = self.max_new_tokens,
            )
        finally:
            inner_lm.generate = original_generate

        self.last_mk_stats = mk_stats

        if not wavs or wavs[0] is None:
            logger.warning("generate_custom_voice returned empty audio — returning silence")
            return np.zeros(TTS_SAMPLE_RATE // 4, dtype=np.float32)  # 250ms silence

        audio = wavs[0]
        return (
            audio.squeeze().astype(np.float32)
            if audio.ndim > 1
            else audio.astype(np.float32)
        )

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """Plain generate_custom_voice() without megakernel (fallback)."""
        wavs, sr = self._qwen_model.generate_custom_voice(
            text           = text,
            language       = self.language,
            speaker        = self.speaker,
            max_new_tokens = self.max_new_tokens,
        )
        self.last_mk_stats = {"fallback": True}
        audio = wavs[0]
        return (
            audio.squeeze().astype(np.float32)
            if audio.ndim > 1
            else audio.astype(np.float32)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        return TTS_SAMPLE_RATE

    @property
    def channels(self) -> int:
        return TTS_CHANNELS
