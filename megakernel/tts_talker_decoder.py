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

Weight layout expected from Qwen3-TTS HF model
───────────────────────────────────────────────
The talker LM lives under `talker` (or `model`) in the state dict.
We detect the key prefix at load time so this works regardless of
how the checkpoint is structured.

Usage
─────
    decoder = TTSTalkerDecoder()          # loads weights + builds kernel
    decoder.prefill(prompt_token_ids)     # fill KV cache with prompt
    for token_id in decoder.stream():     # autoregressive decode, one tok/step
        ...
    decoder.reset()                       # clear KV cache for next utterance
"""

import math
import struct
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
VOCAB_SIZE        = 151936
MAX_SEQ_LEN       = 4096   # Generous for TTS utterances (~1500 codec tokens)


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
        model_name: str = "Qwen/Qwen3-TTS",
        verbose: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self._max_seq_len = max_seq_len
        self._position    = 0

        # ── Build / load extension ────────────────────────────────────────────
        from megakernel.tts_talker_build import get_tts_talker_extension
        ext = get_tts_talker_extension()
        self._decode_op = torch.ops.qwen_tts_talker_C.decode

        # ── Load HF model ─────────────────────────────────────────────────────
        logger.info(f"Loading {model_name} weights…")
        t0 = time.perf_counter()
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

    def _load_weights(self, model_name: str, verbose: bool) -> None:
        """
        Load the talker backbone weights from the Qwen3-TTS HF checkpoint.

        The talker decoder in Qwen3-TTS is a Qwen3-style LM. We extract
        exactly the 11 per-layer tensors the megakernel expects, plus
        embed_tokens, model.norm, and lm_head.

        Key prefix detection handles two layouts seen in practice:
          - "talker.model.layers."  (Qwen3OmniMoeTalker packaging)
          - "model.layers."         (standalone talker checkpoint)
        """
        import os
        from transformers import AutoModel, AutoTokenizer

        token = os.getenv("HF_TOKEN", None)

        # Load the full model in bfloat16 onto CUDA.
        # We only need the talker LM; thinly-wrapped via AutoModel so we get
        # the raw state dict without running any forward pass.
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=token,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, trust_remote_code=True
        )
        state = model.state_dict()

        # Detect which prefix the talker layers live under
        prefix = _find_prefix(state, [
            "talker.model.",    # full Qwen3-TTS checkpoint
            "model.",           # standalone talker checkpoint
        ])

        logger.debug(f"Talker weight prefix detected: '{prefix}'")

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

        # ── Embedding / norm / lm_head ────────────────────────────────────────
        embed_key = f"{prefix}embed_tokens.weight"
        norm_key  = f"{prefix}norm.weight"
        # lm_head may be tied to embed_tokens or a separate key
        lm_key    = "lm_head.weight" if "lm_head.weight" in state else embed_key

        self._embed_weight      = state[embed_key].contiguous()
        self._final_norm_weight = state[norm_key].contiguous()
        self._lm_head_weight    = state[lm_key].contiguous()

        # ── RoPE tables ───────────────────────────────────────────────────────
        self._cos_table, self._sin_table = _build_rope_tables(self._max_seq_len)

        # ── Pack layer weights into the C struct blob ─────────────────────────
        self._layer_weights_packed = _pack_layer_weights(layer_tensors)

        # Keep layer tensors alive (prevent GC of GPU memory)
        self._layer_tensors_ref = layer_tensors

        del model
        torch.cuda.empty_cache()

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

    def step(self, token_id: int) -> int:
        """
        Run one decode step through the megakernel.

        Args:
            token_id: current input token id
        Returns:
            next token id (argmax)
        """
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
        return int(self._out_token.item())

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
