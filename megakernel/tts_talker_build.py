"""
tts_talker_build.py
-------------------
Builds the qwen_megakernel CUDA extension compiled for the TTS talker
(20 layers instead of 28).  We reuse the SAME kernel source from the
qwen_megakernel submodule — no kernel modifications are needed because
num_layers is a runtime parameter passed to launch_ldg_decode_*.

Architectural difference vs. Qwen3-0.6B
----------------------------------------
Component           Qwen3-0.6B      Qwen3-TTS Talker
------------------  --------        ----------------
hidden_size         1024            1024  (same)
num_q_heads         16              16    (same)
num_kv_heads        8               8     (same)
head_dim            128             128   (same)
intermediate_size   3072            3072  (same)
num_hidden_layers   28              20    ← only difference
vocab_size          151936          151936(same, shared tokenizer)

Because every constant except num_layers is identical, the compiled kernel
binary is 100% compatible.  We simply load the extension once and call
decode() with num_layers=20.
"""

import os
import sys

# qwen_megakernel lives as a sibling of e3-assessment:
#   <workspace>/
#   ├── e3-assessment/    ← this repo
#   └── qwen_megakernel/  ← AlpinDale's repo (sibling, separate git)
_HERE       = os.path.dirname(os.path.abspath(__file__))   # e3-assessment/megakernel/
_E3_ROOT    = os.path.dirname(_HERE)                        # e3-assessment/
_WORKSPACE  = os.path.dirname(_E3_ROOT)                     # <workspace>/
_MK_DIR     = os.path.join(_WORKSPACE, "qwen_megakernel")   # <workspace>/qwen_megakernel/
_CSRC       = os.path.join(_MK_DIR, "csrc")

if not os.path.isdir(_CSRC):
    raise RuntimeError(
        f"Cannot find qwen_megakernel/csrc at '{_CSRC}'.\n"
        f"Expected layout:\n"
        f"  <workspace>/\n"
        f"  ├── e3-assessment/   (this repo)\n"
        f"  └── qwen_megakernel/ (AlpinDale's repo — clone it here)\n"
        f"Run: git clone https://github.com/AlpinDale/qwen_megakernel "
        f"'{_MK_DIR}'"
    )

# Make qwen_megakernel importable so we can reuse its build logic
if _MK_DIR not in sys.path:
    sys.path.insert(0, _MK_DIR)

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

    # Same flags as qwen_megakernel/qwen_megakernel/build.py, just a
    # different extension name so both can coexist in the same process.
    KERNEL_FLAGS = [
        f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
        f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
        f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}",
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
