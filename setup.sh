#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-shot build + weight-pull for the RTX 5090 Vast.ai instance
#
# Usage:
#   bash setup.sh          # full setup
#   bash setup.sh --skip-weights  # build only, no HF download
# =============================================================================
set -euo pipefail

SKIP_WEIGHTS=false
for arg in "$@"; do
  [[ "$arg" == "--skip-weights" ]] && SKIP_WEIGHTS=true
done

echo "============================================================"
echo " e3-assessment — RTX 5090 setup"
echo "============================================================"

# ── 0. Sanity checks ─────────────────────────────────────────────────────────
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
GPU=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU detected: $GPU"

# Warn (don't hard-fail) if not a 5090 — kernel might still build on sm_120
if [[ "$GPU" != *"5090"* ]]; then
  echo "WARNING: kernel is tuned for RTX 5090 (sm_120a). On $GPU it will"
  echo "  compile but performance targets may not hold."
fi

CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
echo "CUDA version: $CUDA_VER"

# ── 1. System packages ───────────────────────────────────────────────────────
echo ""
echo "Installing system packages..."
apt-get install -y -q sox libsox-dev ffmpeg 2>/dev/null || \
  echo "  WARNING: apt-get failed (non-root?). Install sox manually if audio I/O errors appear."

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "Installing Python requirements..."
pip install -q -r requirements.txt

# ── 3. Verify transformers >= 4.52.0 (needed for qwen3_tts architecture) ─────
echo ""
echo "Checking transformers version..."
python3 - <<'PYEOF'
import importlib.metadata, sys
from packaging.version import Version

try:
    ver = Version(importlib.metadata.version("transformers"))
except importlib.metadata.PackageNotFoundError:
    ver = Version("0")

MIN_VER = Version("4.52.0")
print(f"  transformers installed: {ver}")

if ver < MIN_VER:
    print(f"  {ver} < {MIN_VER} — qwen3_tts not supported. Installing from source...")
    import subprocess, sys
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/huggingface/transformers.git"
    ])
    new_ver = Version(importlib.metadata.version("transformers"))
    print(f"  Installed from source: {new_ver}")
else:
    print(f"  OK — {ver} >= {MIN_VER}")
PYEOF

# ── 4. Build the megakernel CUDA extension ───────────────────────────────────
echo ""
echo "Pre-building qwen_megakernel CUDA extension (sm_120a)..."

# e3-assessment/ is here; qwen_megakernel/ is the sibling
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
MK_DIR="$WORKSPACE_DIR/qwen_megakernel"

if [[ ! -d "$MK_DIR/csrc" ]]; then
  echo "ERROR: qwen_megakernel not found at $MK_DIR"
  echo "  Clone it first:  git clone https://github.com/AlpinDale/qwen_megakernel $MK_DIR"
  exit 1
fi

python3 - "$MK_DIR" <<'PYEOF'
import sys
mk_dir = sys.argv[1]
sys.path.insert(0, mk_dir)
from qwen_megakernel.build import get_extension
ext = get_extension()
print(f"  Extension built: {ext}")
PYEOF

# Build the TTS talker extension (20-layer variant)
echo ""
echo "Pre-building qwen3_tts_talker CUDA extension (sm_120a, 20 layers)..."
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, '.')
from megakernel.tts_talker_build import get_tts_talker_extension
ext = get_tts_talker_extension()
print(f"  TTS talker extension built: {ext}")
PYEOF

# ── 5. Pull model weights ────────────────────────────────────────────────────
if [[ "$SKIP_WEIGHTS" == "false" ]]; then
  echo ""
  echo "Downloading Qwen3-TTS weights from HuggingFace..."
  echo "(~10 GB — grab a coffee)"
  python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

token = os.getenv("HF_TOKEN", None)

print("  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice ...")
snapshot_download(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    token=token,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print("  Done.")
PYEOF
fi

# ── 6. Copy .env ─────────────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo ""
  echo "Created .env from .env.example — fill in your API keys before running."
fi

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Edit .env with your API keys"
echo "   2. python validate.py          # file-based round-trip test"
echo "   3. python pipeline/run_bot.py  # live WebRTC voice agent"
echo "============================================================"
