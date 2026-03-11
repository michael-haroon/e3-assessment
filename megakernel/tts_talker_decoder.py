"""
tts_talker_decoder.py
---------------------
Stateful megakernel decoder for the Qwen3-TTS talker backbone.

Qwen3-TTS talker architecture vs. Qwen3-0.6B
─────────────────────────────────────────────
Property            Qwen3-0.6B   Qwen3-TTS Talker   Notes
──────────────────  ──────────   ────────────────   ──────────────────────────
hidden_size         1024         1024               identical → kernel reused
num_q_heads         16           16                 identical
num_kv_heads        8            8                  identical
head_dim            128          128                identical
intermediate_size   3072         3072               identical
num_hidden_layers   28           20                 runtime param → no recompile
vocab_size          151936       151936             shared Qwen tokenizer

Because every compile-time constant is identical, we compile the kernel once
and simply pass `num_layers=20` at decode time instead of 28.

Weight layout expected from Qwen3-TTS HF model (confirmed via state dict)
───────────────────────────────────────────────────────────────────────────
The model is NOT a standard Qwen3ForCausalLM. It is a two-vocabulary model:

  talker.model.text_embedding   (151936, 2048)  text token embedding (2048-dim)
  talker.text_projection.*                      2048 → 1024 projection
  talker.model.codec_embedding  (3072, 1024)    codec token embedding (1024-dim)
  talker.model.layers.X.*                       shared 20-layer transformer
  talker.model.norm.weight      (1024,)         final RMS norm
  talker.codec_head.weight      (3072, 1024)    codec output head
  talker.code_predictor.*                       multi-codebook predictor (15 books)

The megakernel handles ONLY the codec autoregressive decode:
  input:  codec token id  (0..3071)
  embed:  talker.model.codec_embedding.weight  row lookup → 1024-dim vector
  decode: 20 transformer layers
  output: talker.codec_head.weight @ hidden    → argmax over 3072 classes

The text conditioning path (text_embedding → text_projection → transformer)
is handled by generate_custom_voice() BEFORE our monkey-patched generate()
is called. By the time _megakernel_generate() receives input_ids, those ids
are CODEC token ids, not text ids.

Usage
─────
    decoder = TTSTalkerDecoder()          # loads weights + builds kernel
    decoder.prefill(prompt_token_ids)     # fill KV cache with prompt
    for token_id in decoder.stream():     # autoregressive decode, one tok/step
        ...
    decoder.reset()                       # clear KV cache for next utterance
"""

import math
import os
import struct
import sys
import time
from typing import Iterator, Optional

import torch
from loguru import logger

# ── Architecture constants (talker backbone) ─────────────────────────────────
NUM_LAYERS        = 20          # Qwen3-TTS talker: 20, not 28
NUM_KV_HEADS      = 8
HEAD_DIM          = 128
HIDDEN_SIZE       = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS       = 16
Q_SIZE            = NUM_Q_HEADS  * HEAD_DIM   # 2048
KV_SIZE           = NUM_KV_HEADS * HEAD_DIM   # 1024

CODEC_VOCAB_SIZE  = 3072    # codec token vocabulary (talker.codec_head output dim)
TEXT_VOCAB_SIZE   = 151936  # text tokenizer vocab  (talker.model.text_embedding)
MAX_SEQ_LEN       = 4096   # Generous for TTS utterances (~1500 codec tokens)
# NOTE: The megakernel handles ONLY the codec token autoregressive decode loop.
#   embed  = talker.model.codec_embedding.weight  (3072, 1024)
#   norm   = talker.model.norm.weight             (1024,)
#   lm_head = talker.codec_head.weight            (3072, 1024)
# Text tokens are encoded by a separate path (text_embedding + text_projection)
# and are NOT part of the megakernel decode path.


# ── Helper: build RoPE tables ─────────────────────────────────────────────────

def _build_rope_tables(max_seq_len: int = MAX_SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def _pack_layer_weights(layer_tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Pack 11 tensors-per-layer into a contiguous byte blob of LDGLayerWeights
    structs (matches the C struct: 11 × void* at 8 bytes each).
    """
    ptr_size   = 8
    n_ptrs     = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_tensors[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


# ── Weight probe helpers ──────────────────────────────────────────────────────

def _find_prefix(state: dict, candidates: list[str]) -> str:
    """Return the first candidate prefix that exists in the state dict."""
    for prefix in candidates:
        probe = f"{prefix}layers.0.input_layernorm.weight"
        if probe in state:
            return prefix
    keys = [k for k in list(state.keys())[:6]]
    raise KeyError(
        f"Cannot find talker layer weights in state dict. "
        f"Tried prefixes {candidates}. First keys: {keys}"
    )


# ─────────────────────────────────────────────────────────────────────────────
class TTSTalkerDecoder:
    """
    Wraps the megakernel for Qwen3-TTS talker autoregressive decode.

    The decoder is stateful: it maintains a KV cache across calls to
    `step()`.  Call `reset()` between utterances.
    """

    def __init__(
        self,
        qwen_model=None,            # Qwen3TTSModel already loaded by pipeline
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        verbose: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self._max_seq_len = max_seq_len
        self._position    = 0

        # ── Build / load extension ────────────────────────────────────────────
        from megakernel.tts_talker_build import get_tts_talker_extension
        get_tts_talker_extension()
        self._decode_op = torch.ops.qwen_tts_talker_C.decode

        # ── Load weights ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if qwen_model is not None:
            logger.info("Extracting talker weights from provided Qwen3TTSModel...")
            self._load_weights_from_model(qwen_model, verbose=verbose)
        else:
            logger.info(f"Loading {model_name} weights from HuggingFace...")
            self._load_weights(model_name, verbose=verbose)
        logger.info(f"Weights loaded in {time.perf_counter() - t0:.1f}s")

        # ── KV cache ──────────────────────────────────────────────────────────
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # ── Scratch buffers (single-token decode) ─────────────────────────────
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden    = torch.empty(HIDDEN_SIZE, **bf16)
        self._act       = torch.empty(HIDDEN_SIZE, **f32)
        self._res       = torch.empty(HIDDEN_SIZE, **f32)
        self._q         = torch.empty(Q_SIZE, **f32)
        self._k         = torch.empty(KV_SIZE, **f32)
        self._v         = torch.empty(KV_SIZE, **f32)
        self._attn_out  = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_weights_from_model(self, qwen_model, verbose: bool) -> None:
        """
        Extract talker backbone weights from an already-loaded Qwen3TTSModel.

        Avoids a second checkpoint download when the pipeline has already
        called Qwen3TTSModel.from_pretrained().  We pull state_dict() from
        qwen_model.model (the inner HF backbone) so the prefix detection
        logic is shared with the standalone path below.

        The tokenizer is also reused from the qwen_model to avoid a second
        HF network call.
        """
        inner = qwen_model.model          # HF backbone (Qwen3 transformer)
        state = inner.state_dict()

        prefix = _find_prefix(state, [
            "talker.model.",    # Qwen3OmniMoeTalker packaging
            "model.",           # standalone / custom-voice layout
            "",                 # flat state dict (embed_tokens.weight at root)
        ])
        logger.debug(f"Talker weight prefix (from model): '{prefix}'")
        self._extract_from_state(state, prefix)

        # Reuse tokenizer already bundled inside qwen_model
        self._tokenizer = getattr(qwen_model, "tokenizer", None)

    def _load_weights(self, model_name: str, verbose: bool) -> None:
        """
        Load the talker backbone weights from the Qwen3-TTS HF checkpoint.

        Fallback path used when no pre-loaded Qwen3TTSModel is provided.
        Uses qwen_tts.Qwen3TTSModel (not AutoModel) because qwen3_tts is not
        a registered HF architecture in standard transformers releases.
        """
        import os
        from qwen_tts import Qwen3TTSModel

        token = os.getenv("HF_TOKEN", None)

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",   # cuDNN FA3 on Blackwell; no flash-attn pkg needed
        )
        # Reuse tokenizer bundled in Qwen3TTSModel
        self._tokenizer = getattr(model, "tokenizer", None)

        inner = model.model          # HF backbone (Qwen3 transformer)
        state = inner.state_dict()

        prefix = _find_prefix(state, [
            "talker.model.",
            "model.",
            "",
        ])
        logger.debug(f"Talker weight prefix detected: '{prefix}'")
        self._extract_from_state(state, prefix)

        del model
        torch.cuda.empty_cache()

    def _extract_from_state(self, state: dict, prefix: str) -> None:
        """
        Shared weight-extraction logic used by both load paths.



        Qwen3-TTS talker weight layout (confirmed from state dict inspection):

          talker.model.layers.X.*                 ← transformer layers (prefix = 'talker.model.')
          talker.model.norm.weight      (1024,)   ← final RMS norm
          talker.model.codec_embedding.weight     ← codec token embedding (3072, 1024)
          talker.codec_head.weight      (3072, 1024) ← output projection (ONE level above prefix)

        text_embedding (151936, 2048) and text_projection are NOT used here;
        those are the conditioning path handled by generate_custom_voice().

        Key lookup strategy for embed / norm / lm_head:
          - prefix is where layers live, e.g. 'talker.model.'
          - norm  is at {prefix}norm.weight               → always same level as layers
          - embed is at {prefix}codec_embedding.weight    → same level as layers
          - lm_head is ONE level above prefix             → parent_prefix + 'codec_head.weight'
        """
        # ── Per-layer tensors (11 per layer) ──────────────────────────────────
        layer_tensors = []
        for i in range(NUM_LAYERS):
            p = f"{prefix}layers.{i}."
            layer_tensors.extend([
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ])

        # ── norm: always at same level as layers ──────────────────────────────
        norm_key = f"{prefix}norm.weight"
        if norm_key not in state:
            raise KeyError(f"norm not found at '{norm_key}'. State has {len(state)} keys.")

        # ── embed: codec_embedding is at same level as layers ─────────────────
        # (text_embedding is 2048-dim and goes through text_projection — not used here)
        embed_candidates = [
            f"{prefix}codec_embedding.weight",   # Qwen3-TTS CustomVoice
            f"{prefix}embed_tokens.weight",      # standard Qwen3 LM layout
        ]
        embed_key = next((k for k in embed_candidates if k in state), None)
        if embed_key is None:
            raise KeyError(
                f"Cannot find codec embedding near prefix '{prefix}'. "
                f"Tried: {embed_candidates}"
            )
        logger.debug(f"Using embed key: {embed_key}")

        # ── lm_head: one level above prefix (talker.codec_head, not talker.model.lm_head) ──
        # Compute parent prefix by stripping the last path component from prefix.
        # e.g. 'talker.model.' -> parent = 'talker.'
        parent_prefix = ""
        stripped = prefix.rstrip(".")
        if "." in stripped:
            parent_prefix = stripped.rsplit(".", 1)[0] + "."

        lm_candidates = [
            f"{parent_prefix}codec_head.weight",   # Qwen3-TTS: talker.codec_head.weight
            "lm_head.weight",                      # standard: root-level
            f"{prefix}lm_head.weight",             # same level as layers
            embed_key,                             # tied weights fallback
        ]
        lm_key = next((k for k in lm_candidates if k in state), None)
        if lm_key is None:
            raise KeyError(
                f"Cannot find lm_head near prefix '{prefix}'. Tried: {lm_candidates}"
            )
        logger.debug(f"Using lm_head key: {lm_key}")

        self._embed_weight      = state[embed_key].contiguous()
        self._final_norm_weight = state[norm_key].contiguous()
        self._lm_head_weight    = state[lm_key].contiguous()

        logger.debug(
            f"Weight shapes — embed: {tuple(self._embed_weight.shape)}, "
            f"norm: {tuple(self._final_norm_weight.shape)}, "
            f"lm_head: {tuple(self._lm_head_weight.shape)}"
        )

        # ── RoPE tables ───────────────────────────────────────────────────────
        self._cos_table, self._sin_table = _build_rope_tables(self._max_seq_len)

        # ── Pack layer weights into the C struct blob ─────────────────────────
        self._layer_weights_packed = _pack_layer_weights(layer_tensors)

        # Keep layer tensors alive (prevent GC of GPU memory)
        self._layer_tensors_ref = layer_tensors

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear KV cache and position counter. Call between utterances."""
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def codec_vocab_size(self) -> int:
        return CODEC_VOCAB_SIZE

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def step(self, token_id: int) -> int:
        """
        Run one decode step through the megakernel.

        Args:
            token_id: current input token id
        Returns:
            next token id (argmax)
        """
        if token_id < 0 or token_id >= CODEC_VOCAB_SIZE:
            raise ValueError(
                f"Codec token id out of range: {token_id} (expected 0..{CODEC_VOCAB_SIZE-1})"
            )
        if self._position >= self._max_seq_len:
            raise RuntimeError(
                f"Megakernel sequence length exceeded: pos={self._position}, max={self._max_seq_len}"
            )

        self._decode_op(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,            # ← 20 for TTS talker, not 28
            self._position,
            self._max_seq_len,
            self._attn_scale,
        )
        self._position += 1
        next_token = int(self._out_token.item())
        if next_token < 0 or next_token >= CODEC_VOCAB_SIZE:
            raise RuntimeError(
                f"Megakernel produced invalid token id {next_token} at position {self._position}"
            )
        return next_token

    def prefill(self, token_ids: list[int]) -> None:
        """
        Feed prompt tokens into the KV cache without keeping output tokens.
        After prefill the KV cache is primed for autoregressive decode.
        """
        for tid in token_ids[:-1]:
            self.step(tid)
        # Return the last prefill token so callers can use it as first decode input
        self._prefill_last = token_ids[-1] if token_ids else 0

    def stream(
        self,
        first_token_id: int,
        eos_token_ids: set[int],
        max_new_tokens: int = 1500,
        temperature: float = 1.0,   # reserved; kernel is argmax-only
    ) -> Iterator[int]:
        """
        Autoregressive decode generator.  Yields token ids one at a time.

        Args:
            first_token_id: token to start decoding from (e.g. last prefill token)
            eos_token_ids:  set of token ids that signal end of sequence
            max_new_tokens: hard cap
        Yields:
            int token ids
        """
        token_id = first_token_id
        for _ in range(max_new_tokens):
            token_id = self.step(token_id)
            yield token_id
            if token_id in eos_token_ids:
                break
