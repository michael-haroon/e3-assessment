"""
tts_talker_build.py
-------------------
Builds the qwen_megakernel CUDA extension compiled for the TTS talker
(20 layers instead of 28, vocab_size=3072 instead of 151936).

Source: the local megakernel/ directory (e3-assessment/megakernel/),
which contains kernel.cu compiled with LDG_VOCAB_SIZE=3072.

Architectural difference vs. Qwen3-0.6B
----------------------------------------
Component           Qwen3-0.6B      Qwen3-TTS Talker
------------------  --------        ----------------
hidden_size         1024            1024  (same)
num_q_heads         16              16    (same)
num_kv_heads        8               8     (same)
head_dim            128             128   (same)
intermediate_size   3072            3072  (same)
num_hidden_layers   28              20    ← runtime arg
vocab_size          151936          3072  ← compile-time flag
"""

import os
import sys

# Build from the local megakernel/ directory (this file's directory).
# kernel.cu here is compiled with LDG_VOCAB_SIZE=3072 (TTS codec vocab).
_HERE  = os.path.dirname(os.path.abspath(__file__))   # e3-assessment/megakernel/
_CSRC  = _HERE                                          # kernel.cu / torch_bindings.cpp live here

if not os.path.isfile(os.path.join(_CSRC, "kernel.cu")):
    raise RuntimeError(
        f"Cannot find kernel.cu at '{_CSRC}'.\n"
        f"Expected e3-assessment/megakernel/kernel.cu to exist."
    )

_tts_talker_ext = None


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


def get_tts_talker_extension():
    """
    Return (and lazily build) the CUDA extension for Qwen3-TTS talker decode.

    We compile with the same flags as the original megakernel.  The only
    runtime difference is that callers pass num_layers=20 instead of 28.
    """
    global _tts_talker_ext
    if _tts_talker_ext is not None:
        return _tts_talker_ext

    from torch.utils.cpp_extension import load

    # Build flags for the TTS talker kernel (20 layers, LM head scaled to
    # codec vocab). LDG_VOCAB_SIZE is hardcoded as a constexpr in kernel.cu
    # (line 74) so no compile flag is needed here.
    # LDG_LM_NUM_BLOCKS=12 is scaled down from 1280 proportionally to the
    # codec vocab size (3072 vs 151936).
    KERNEL_FLAGS = [
        f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
        f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
        f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 12)}",
        f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
        f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
        f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
        f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
        f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
        f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
        f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
        f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
        f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
        f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
        "-DLDG_USE_UINT4",
        "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS",
        "-DLDG_MLP_SMEM",
    ]

    CUDA_FLAGS = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-arch=sm_120a",
        f"-I{_CSRC}",
    ] + KERNEL_FLAGS

    _tts_talker_ext = load(
        name="qwen_tts_talker_C",          # distinct name from base megakernel
        sources=[
            os.path.join(_CSRC, "torch_bindings.cpp"),
            os.path.join(_CSRC, "kernel.cu"),
        ],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _tts_talker_ext
