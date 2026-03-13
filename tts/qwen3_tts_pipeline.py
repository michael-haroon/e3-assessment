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
        self._generation_kwargs: dict = {}

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
        inner_lm = self._qwen_model.model.talker
        original_generate = inner_lm.generate
        mk_stats: dict = {}

        def _megakernel_generate(inputs_embeds=None, input_ids=None, **kwargs):
            max_tokens = min(
                kwargs.get("max_new_tokens", self.max_new_tokens),
                _MAX_TOKENS_HARD_CAP,
            )

            # ── Step 1: HF prefill (1 step) to get text-prefix KV cache ──────
            prefill_kwargs = {
                **kwargs,
                "max_new_tokens": 1,
                "min_new_tokens": 1,   # prevent conflict with talker's min_new_tokens=2
                "return_dict_in_generate": True,
                "use_cache": True,
                "output_hidden_states": True,
            }
            with torch.no_grad():
                prefill_out = original_generate(
                    inputs_embeds=inputs_embeds, **prefill_kwargs
                )

            first_token = int(prefill_out.sequences[0, 0].item())
            past_kv     = prefill_out.past_key_values
            N_prefix    = past_kv.get_seq_length()

            # ── Step 2: Copy HF KV cache → kernel ────────────────────────────
            # New DynamicCache API (transformers ≥ 4.46 layer-per-object style):
            #   past_kv.layers[i].keys   shape [1, num_kv_heads, seq_len, head_dim]
            # Kernel KV shape: [num_layers, num_kv_heads, max_seq_len, head_dim]
            #
            # IMPORTANT: past_kv has N_prefix = N_text + 1 positions.
            # The +1 is from HF's one decode step that already processed
            # first_token (putting its KV at position N_text).
            # We copy only the N_text text-context positions so the kernel
            # starts clean; step(first_token) then fills position N_text
            # exactly as HF decode step 0 did — no double-processing.
            from megakernel.tts_talker_decoder import NUM_LAYERS
            N_text = N_prefix - 1   # strip the one HF decode step
            self._talker.reset()
            for layer_idx in range(NUM_LAYERS):
                layer = past_kv.layers[layer_idx]   # DynamicLayer
                k = layer.keys    # [1, 8, N_prefix, 128]
                v = layer.values
                self._talker._k_cache[layer_idx, :, :N_text, :] = k[0, :, :N_text, :]
                self._talker._v_cache[layer_idx, :, :N_text, :] = v[0, :, :N_text, :]
            self._talker._position = N_text

            # past_hidden for first code_predictor call = last layer output
            # from the HF prefill step (hidden_states[0] = prefill forward).
            # hidden_states[0][0][-1][:, -1:] = last layer, last position.
            past_hidden = (
                prefill_out.hidden_states[0][0][-1][:, -1:].detach()
            )  # [1, 1, 1024]

            # ── Step 3: EOS set ───────────────────────────────────────────────
            raw_eos = kwargs.get("eos_token_id", [])
            if isinstance(raw_eos, int):
                raw_eos = [raw_eos]
            elif isinstance(raw_eos, torch.Tensor):
                raw_eos = raw_eos.tolist()
            eos_set = set(int(x) for x in raw_eos if x is not None)

            # ── Step 4: helpers ────────────────────────────────────────────────
            dummy_hs = torch.zeros(
                1, 1, inner_lm.config.hidden_size,
                dtype=torch.bfloat16, device="cuda",
            )
            num_code_groups = inner_lm.config.num_code_groups  # 16
            codec_embed_fn  = inner_lm.get_input_embeddings()  # codec_embedding
            cp_embed_fns    = inner_lm.code_predictor.get_input_embeddings()  # ModuleList[15]

            # Text conditioning passed from outer model.generate()
            trailing_text_hidden = kwargs.get("trailing_text_hidden")  # [1, T, H]
            tts_pad_embed        = kwargs.get("tts_pad_embed")         # [1, 1, H]

            suppress_list = kwargs.get("suppress_tokens", [])
            suppress_t = (
                torch.tensor(suppress_list, dtype=torch.long, device="cuda")
                if suppress_list else None
            )

            def _run_predictor(cb0_token, ph):
                """Predict cb1..cb15 given cb0 and past_hidden. Returns [1, 16]."""
                cb0_id = torch.tensor([[cb0_token]], dtype=torch.long, device="cuda")
                cb0_embed = codec_embed_fn(cb0_id)  # [1, 1, H]
                with torch.no_grad():
                    pred = inner_lm.code_predictor.generate(
                        inputs_embeds=torch.cat((ph, cb0_embed), dim=1),
                        max_new_tokens=num_code_groups - 1,
                        do_sample=kwargs.get("subtalker_dosample", True),
                        top_p=kwargs.get("subtalker_top_p", 1.0),
                        temperature=kwargs.get("subtalker_temperature", 0.9),
                        return_dict_in_generate=True,
                        use_cache=True,
                    )
                return torch.cat([cb0_id, pred.sequences], dim=-1)  # [1, 16]

            def _combined_embed(cb0_token, codec_ids, gen_step):
                """
                Replicate the HF talker's input embedding:
                  sum(codec_embed(cb0), cp_embed[0](cb1), ..., cp_embed[14](cb15))
                  + trailing_text_hidden[gen_step]  (or tts_pad_embed)
                Returns [H] bfloat16 — ready to inject into _embed_weight row.
                """
                cb0_e = codec_embed_fn(codec_ids[:, 0:1])  # [1, 1, H]
                parts = [cb0_e]
                for i in range(num_code_groups - 1):
                    parts.append(cp_embed_fns[i](codec_ids[:, i + 1 : i + 2]))
                codec_sum = torch.stack(parts, dim=0).sum(0)  # [1, 1, H]
                if gen_step < trailing_text_hidden.shape[1]:
                    text_cond = trailing_text_hidden[:, gen_step : gen_step + 1]
                else:
                    text_cond = tts_pad_embed
                combined = codec_sum + text_cond  # [1, 1, H]
                return combined[0, 0]  # [H] bfloat16

            def _kernel_step_with_embed(token_id, embed_vec):
                """Run kernel step() but use embed_vec instead of plain lookup."""
                orig = self._talker._embed_weight[token_id].clone()
                self._talker._embed_weight[token_id] = embed_vec
                self._talker.step(token_id)
                self._talker._embed_weight[token_id] = orig

            def _next_token_from_logits():
                """Compute next cb0 from _norm_out with suppress_tokens."""
                logits = (
                    self._talker._norm_out.float()
                    @ self._talker._lm_head_weight.float().T
                )
                if suppress_t is not None:
                    logits.scatter_(0, suppress_t, float("-inf"))
                return int(logits.argmax().item())

            # ── Step 5 + 6: Decode loop ───────────────────────────────────────
            # At each step the code_predictor runs FIRST (cb1..cb15 are part of
            # the combined input embedding), then the kernel processes the
            # combined embedding through the 28-layer transformer.
            all_codec_ids = []
            all_tokens    = []
            current_cb0   = first_token
            ph            = past_hidden   # from HF prefill
            gen_step      = 0

            t0 = time.perf_counter()
            for _ in range(max_tokens):
                all_tokens.append(current_cb0)

                # a) code_predictor → cb1..cb15 for this cb0
                cids = _run_predictor(current_cb0, ph)
                all_codec_ids.append(cids)

                # b) stop if EOS (codec_ids already recorded for outer stripping)
                if current_cb0 in eos_set:
                    break

                # c) combined embed → kernel step → next token
                embed_vec = _combined_embed(current_cb0, cids, gen_step)
                _kernel_step_with_embed(current_cb0, embed_vec)
                current_cb0 = _next_token_from_logits()

                # d) past_hidden for next code_predictor = kernel's normed output
                ph = (
                    self._talker._norm_out
                    .detach().clone()
                    .to(torch.bfloat16)
                    .view(1, 1, -1)
                )
                gen_step += 1
            elapsed = time.perf_counter() - t0

            mk_stats.update(
                tokens=len(all_tokens),
                elapsed=elapsed,
                tok_per_s=len(all_tokens) / elapsed if elapsed > 0 else 0,
                hit_eos=bool(eos_set and all_tokens[-1] in eos_set),
                generated_tokens=list(all_tokens),
            )
            logger.info(
                f"Megakernel: {len(all_tokens)} tok in {elapsed*1000:.1f}ms "
                f"({mk_stats['tok_per_s']:.0f} tok/s) "
                f"EOS={'✅' if mk_stats['hit_eos'] else '❌ hit token limit'}"
            )

            # ── Step 7: Build fake GenerateOutput for model.generate() ────────
            # Outer generate (Qwen3TTSForConditionalGeneration.generate) reads:
            #   hid[-1]    → codec_ids [1, 16]         (used for audio)
            #   hid[0][-1] → last layer hidden [1,1,H]  (discarded via _)
            class _FakeOut:
                sequences     = torch.tensor(
                    [all_tokens], dtype=torch.long, device="cuda"
                )
                hidden_states = tuple(
                    ((dummy_hs,), cids) for cids in all_codec_ids
                )  # materialised tuple so outer code can iterate it twice

            return _FakeOut()

        inner_lm.generate = _megakernel_generate
        try:
            wavs, sr = self._qwen_model.generate_custom_voice(
                text           = text,
                language       = self.language,
                speaker        = self.speaker,
                max_new_tokens = self.max_new_tokens,
                **self._generation_kwargs,
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
            **self._generation_kwargs,
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
