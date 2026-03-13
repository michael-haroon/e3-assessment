"""Global import-time shims used across entry points.

Must be imported before any ``qwen_tts`` import in the process.

This module is intentionally idempotent and side-effect-only.
"""

import importlib
import sys


def _alias_module(dst: str, src: str) -> bool:
    """Alias an importable module ``src`` under module name ``dst``."""
    if dst in sys.modules:
        return True
    try:
        mod = importlib.import_module(src)
    except Exception:
        return False
    sys.modules[dst] = mod
    return True


def _install_flash_attn_aliases() -> None:
    """Expose whichever flash attention package is installed as ``flash_attn``."""
    if "flash_attn" in sys.modules:
        return

    # Prefer canonical package if present.
    if _alias_module("flash_attn", "flash_attn"):
        _alias_module("flash_attn.flash_attn_interface", "flash_attn.flash_attn_interface")
        return

    # Fallback: some environments provide FlashAttention v3 under flash_attn_3.
    if _alias_module("flash_attn", "flash_attn_3"):
        _alias_module("flash_attn.flash_attn_interface", "flash_attn_3.flash_attn_interface")


_install_flash_attn_aliases()
