"""
inspect_state_dict.py
---------------------
Dumps the state dict key structure of Qwen3TTSModel so we can
map the correct paths for embed_tokens, norm, and lm_head.

Run:
    python inspect_state_dict.py

Output: grouped key listing showing shape of every unique "leaf" name.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from qwen_tts import Qwen3TTSModel

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

print(f"Loading {MODEL} ...")
m = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cpu",           # cpu so we don't waste VRAM
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

inner = m.model
state = inner.state_dict()

print(f"\nTotal keys in qwen_model.model.state_dict(): {len(state)}\n")

# ── Print every key that contains embed / norm / lm_head / talker ─────────
INTERESTING = ("embed", "norm", "lm_head", "talker")
print("── Keys matching embed / norm / lm_head / talker ──")
for k, v in state.items():
    if any(tok in k for tok in INTERESTING) and "layers." not in k:
        print(f"  {k:<70s}  {tuple(v.shape)}")

# ── Print first 30 keys (raw) to see overall layout ───────────────────────
print("\n── First 30 keys (raw) ──")
for k in list(state.keys())[:30]:
    print(f"  {k}")

# ── Print last 10 keys ────────────────────────────────────────────────────
print("\n── Last 10 keys (raw) ──")
for k in list(state.keys())[-10:]:
    print(f"  {k}")
