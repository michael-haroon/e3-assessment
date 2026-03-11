"""
bootstrap.py
────────────
Must be imported before any qwen_tts import anywhere in the process.

1. Shims flash_attn_3 → flash_attn so qwen_tts's import-time check passes
   without the "flash-attn is not installed" warning.
2. Safe to import multiple times (idempotent).
"""

import sys

if "flash_attn" not in sys.modules:
    try:
        import flash_attn_3 as _fa3
        sys.modules["flash_attn"] = _fa3

        try:
            import flash_attn_3.flash_attn_interface as _fa3_iface
            sys.modules["flash_attn.flash_attn_interface"] = _fa3_iface
        except ImportError:
            pass

    except ImportError:
        pass  # neither package present — qwen_tts will warn as normal
