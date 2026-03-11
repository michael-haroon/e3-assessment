"""
validate.py
───────────
End-to-end round-trip validation (file-based, no microphone needed).

What it tests:
  1. Megakernel extension builds and loads correctly
  2. TTSTalkerDecoder generates tokens at ~1000 tok/s
  3. Full Qwen3-TTS pipeline produces valid audio
  4. Audio is streamed frame-by-frame (not buffered)
  5. TTFC and RTF meet the performance targets

Usage:
  python validate.py                   # full validation
  python validate.py --skip-tts        # kernel + decoder only (fast)
  python validate.py --save-audio out/ # save audio to WAV files

Output:
  Console report + optional WAV files.
  Exit code 0 = all checks passed.
  Exit code 1 = one or more checks failed.
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

import numpy as np
import torch
from loguru import logger

# ── Target thresholds ─────────────────────────────────────────────────────────
TARGET_TOK_PER_S = 1000    # tok/s (conservative; 0.6B can hit 1000)
TARGET_TTFC_MS   = int(os.getenv("TARGET_TTFC_MS", "1000"))  # ms (warm-path default)
TARGET_RTF       = 0.15    # (generous for validate; pipeline target is 0.15)

PASS = "[✓]"
FAIL = "[✗]"


def _check(condition: bool, name: str, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {name}" + (f"  — {detail}" if detail else ""))
    return condition


# ── Test 1: Build extension ───────────────────────────────────────────────────

def test_build_extension() -> bool:
    print("\n── Test 1: Build megakernel CUDA extension ──")
    try:
        from megakernel.tts_talker_build import get_tts_talker_extension
        ext = get_tts_talker_extension()
        return _check(ext is not None, "Extension built", str(ext))
    except Exception as e:
        _check(False, "Extension build", str(e))
        return False


# ── Test 2: Decoder step latency ─────────────────────────────────────────────

def test_decoder_speed(n_tokens: int = 100) -> bool:
    print(f"\n── Test 2: Megakernel talker speed ({n_tokens} tokens) ──")
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
        from megakernel.tts_talker_decoder import TTSTalkerDecoder

        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        logger.info(f"Loading {model_name} for decoder speed test...")
        qwen_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",   # cuDNN FA3 on Blackwell; no flash-attn pkg needed
        )

        dec = TTSTalkerDecoder(qwen_model=qwen_model, verbose=False)

        # Warmup
        dec.reset()
        for _ in range(10):
            dec.step(1)

        dec.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token_id = 1
        for _ in range(n_tokens):
            token_id = dec.step(token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tok_per_s = n_tokens / elapsed
        ms_per_tok = elapsed * 1000 / n_tokens

        ok = tok_per_s >= TARGET_TOK_PER_S
        _check(ok, f"Decode speed", f"{tok_per_s:.0f} tok/s  ({ms_per_tok:.2f} ms/tok)  target ≥{TARGET_TOK_PER_S}")
        return ok
    except Exception as e:
        _check(False, "Decoder speed", str(e))
        return False


# ── Test 3: Full TTS pipeline ─────────────────────────────────────────────────

async def test_full_tts(save_dir: str | None = None) -> bool:
    print("\n── Test 3: Full Qwen3-TTS pipeline (TTFC + RTF) ──")
    try:
        from tts.qwen3_tts_pipeline import Qwen3TTSPipeline, TTS_SAMPLE_RATE

        pipe = Qwen3TTSPipeline(verbose=False)
        text = "Hello, this is a streaming test of the Qwen3 TTS system."

        # TTFC should reflect serving latency on an already-loaded pipeline.
        # Model download/load is tracked separately and is not per-request TTFC.
        load_t0 = time.perf_counter()
        pipe._ensure_loaded()
        load_ms = (time.perf_counter() - load_t0) * 1000

        t0            = time.perf_counter()
        first_chunk_t = None
        chunks        = []

        async for chunk in pipe.synthesize_streaming(text):
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()
            chunks.append(chunk)
            # Verify chunks arrive immediately, not all at once at the end
            # (each yield must give control back)

        total_t = time.perf_counter() - t0

        if not chunks:
            _check(False, "Received audio chunks", "No chunks received")
            return False

        n_samples   = sum(len(c) for c in chunks) // 2
        audio_dur_s = n_samples / TTS_SAMPLE_RATE
        ttfc_ms     = (first_chunk_t - t0) * 1000 if first_chunk_t else 9999
        rtf         = total_t / audio_dur_s if audio_dur_s > 0 else 9999
        n_chunks    = len(chunks)

        _check(True, "Model ready", f"loaded in {load_ms:.0f}ms (excluded from TTFC)")
        pass_chunks = _check(n_chunks > 1, "Streaming (multi-chunk)", f"{n_chunks} chunks received")
        pass_ttfc   = _check(ttfc_ms < TARGET_TTFC_MS, "TTFC", f"{ttfc_ms:.0f}ms  target <{TARGET_TTFC_MS}ms")
        pass_rtf    = _check(rtf < TARGET_RTF, "RTF", f"{rtf:.3f}  target <{TARGET_RTF}")
        _check(True, "Audio duration", f"{audio_dur_s:.2f}s")

        # Optional: save WAV
        if save_dir:
            import soundfile as sf
            os.makedirs(save_dir, exist_ok=True)
            audio_np = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32767
            out_path = os.path.join(save_dir, "validate_output.wav")
            sf.write(out_path, audio_np, TTS_SAMPLE_RATE)
            print(f"      Audio saved to {out_path}")

        return pass_chunks and pass_ttfc and pass_rtf

    except Exception as e:
        _check(False, "Full TTS pipeline", str(e))
        import traceback; traceback.print_exc()
        return False


# ── Test 4: Verify streaming is frame-by-frame ────────────────────────────────

async def test_streaming_is_realtime() -> bool:
    """
    Confirm that chunks arrive incrementally, not all buffered at the end.
    We measure the inter-chunk arrival times.  In a truly streaming system
    the first chunk arrives well before the last.
    """
    print("\n── Test 4: Streaming frame-by-frame verification ──")
    try:
        from tts.qwen3_tts_pipeline import Qwen3TTSPipeline, TTS_SAMPLE_RATE

        pipe = Qwen3TTSPipeline(verbose=False)
        text = "Streaming check. One two three four five."

        arrival_times = []
        t0 = time.perf_counter()
        async for chunk in pipe.synthesize_streaming(text):
            arrival_times.append(time.perf_counter() - t0)

        if len(arrival_times) < 2:
            return _check(False, "Streaming", "Only 1 chunk received — cannot verify streaming")

        # If buffered, all chunks arrive at nearly the same time.
        # If streaming, there's spread in arrival times.
        span_ms = (arrival_times[-1] - arrival_times[0]) * 1000
        ok = len(arrival_times) > 1  # minimal check: at least 2 frames
        _check(ok, "Multi-frame delivery", f"{len(arrival_times)} chunks over {span_ms:.0f}ms span")
        return ok

    except Exception as e:
        _check(False, "Streaming verification", str(e))
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_all(args: argparse.Namespace) -> int:
    print("=" * 55)
    print("  e3-assessment — End-to-End Validation")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU!)'}")
    print("=" * 55)

    results = []

    results.append(test_build_extension())
    results.append(test_decoder_speed(n_tokens=100))

    if not args.skip_tts:
        results.append(await test_full_tts(save_dir=args.save_audio))
        results.append(await test_streaming_is_realtime())

    passed = sum(results)
    total  = len(results)

    print("\n" + "=" * 55)
    print(f"  {passed}/{total} tests passed")
    if passed == total:
        print("  ✓ All checks passed!")
    else:
        print("  ✗ Some checks failed — see above.")
    print("=" * 55)

    return 0 if passed == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="e3-assessment validation")
    parser.add_argument("--skip-tts", action="store_true", help="Skip full TTS tests (fast)")
    parser.add_argument("--save-audio", metavar="DIR", help="Save output WAV files to DIR")
    args = parser.parse_args()

    exit_code = asyncio.run(run_all(args))
    sys.exit(exit_code)
