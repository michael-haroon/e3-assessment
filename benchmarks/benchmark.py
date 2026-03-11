"""
benchmark.py
─────────────
Standalone benchmark: measures megakernel talker throughput, TTFC, and RTF
without running the full Pipecat pipeline.

Usage:
  python benchmarks/benchmark.py                  # full suite
  python benchmarks/benchmark.py --quick          # 3 utterances, no warmup
  python benchmarks/benchmark.py --no-megakernel  # HF baseline only

Output:
  Rich table with per-run numbers + aggregate summary.
  Also writes results to benchmarks/results.json for CI / README generation.
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

# ── Test utterances ───────────────────────────────────────────────────────────
SHORT_UTTERANCES = [
    "Hello, how are you today?",
    "The weather is nice.",
    "I enjoy working on AI projects.",
]

LONG_UTTERANCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming many industries including healthcare and finance.",
    "Please let me know if there is anything else I can help you with today.",
]


# ── Megakernel talker benchmark ───────────────────────────────────────────────

def bench_megakernel_talker(
    n_warmup: int = 2,
    n_runs: int = 5,
    max_tokens: int = 100,
) -> dict:
    """
    Benchmark the megakernel talker in isolation (LM decode only, no vocoder).
    Uses dummy token ids to avoid needing a real TTS prompt.
    """
    from megakernel.tts_talker_decoder import TTSTalkerDecoder, NUM_LAYERS, VOCAB_SIZE

    console.print("[cyan]Loading TTSTalkerDecoder…[/cyan]")
    try:
        decoder = TTSTalkerDecoder(verbose=False)
    except Exception as e:
        console.print(f"[red]TTSTalkerDecoder load failed: {e}[/red]")
        return {"error": str(e)}

    # Warmup
    console.print(f"[cyan]Warming up ({n_warmup} runs × {max_tokens} tokens)…[/cyan]")
    for _ in range(n_warmup):
        decoder.reset()
        token_id = 1  # dummy BOS-like token
        for _ in range(max_tokens):
            token_id = decoder.step(token_id)
    torch.cuda.synchronize()

    # Timed runs
    console.print(f"[cyan]Benchmarking ({n_runs} runs × {max_tokens} tokens)…[/cyan]")
    run_times = []
    for _ in range(n_runs):
        decoder.reset()
        token_id = 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max_tokens):
            token_id = decoder.step(token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)

    avg_elapsed = sum(run_times) / len(run_times)
    tok_per_s   = max_tokens / avg_elapsed
    ms_per_tok  = avg_elapsed * 1000 / max_tokens

    return {
        "backend":     "Megakernel (talker, 20 layers)",
        "tokens":      max_tokens,
        "runs":        n_runs,
        "avg_elapsed": avg_elapsed,
        "tok_per_s":   tok_per_s,
        "ms_per_tok":  ms_per_tok,
        "run_times":   run_times,
    }


# ── HF baseline benchmark ─────────────────────────────────────────────────────

def bench_hf_baseline(
    n_warmup: int = 1,
    n_runs: int = 3,
    max_tokens: int = 100,
) -> dict:
    """Benchmark Qwen3-0.6B via HuggingFace generate() as baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print("[cyan]Loading HF baseline (Qwen/Qwen3-0.6B)…[/cyan]")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
    except Exception as e:
        console.print(f"[red]HF baseline load failed: {e}[/red]")
        return {"error": str(e)}

    input_ids = tokenizer("Hello", return_tensors="pt").input_ids.cuda()

    def _run():
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )

    for _ in range(n_warmup):
        _run()
    torch.cuda.synchronize()

    run_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run()
        torch.cuda.synchronize()
        run_times.append(time.perf_counter() - t0)

    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    avg = sum(run_times) / len(run_times)
    return {
        "backend":     "HuggingFace generate() (Qwen3-0.6B, baseline)",
        "tokens":      max_tokens,
        "runs":        n_runs,
        "avg_elapsed": avg,
        "tok_per_s":   max_tokens / avg,
        "ms_per_tok":  avg * 1000 / max_tokens,
        "run_times":   run_times,
    }


# ── Full TTS pipeline benchmark (TTFC + RTF) ──────────────────────────────────

async def bench_full_tts(utterances: list[str]) -> list[dict]:
    """
    Benchmark end-to-end TTS: text → first audio chunk (TTFC) and RTF.
    """
    from tts.qwen3_tts_pipeline import Qwen3TTSPipeline, TTS_SAMPLE_RATE

    console.print("[cyan]Loading Qwen3TTSPipeline…[/cyan]")
    pipe = Qwen3TTSPipeline(verbose=False)

    results = []
    for i, text in enumerate(utterances):
        console.print(f"[cyan]  [{i+1}/{len(utterances)}] '{text[:50]}…'[/cyan]")

        t0 = time.perf_counter()
        first_chunk_t = None
        total_bytes   = 0

        async for chunk in pipe.synthesize_streaming(text):
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()
            total_bytes += len(chunk)

        total_t = time.perf_counter() - t0

        n_samples   = total_bytes // 2         # int16
        audio_dur_s = n_samples / TTS_SAMPLE_RATE
        ttfc_ms     = (first_chunk_t - t0) * 1000 if first_chunk_t else 0
        rtf         = total_t / audio_dur_s if audio_dur_s > 0 else 0

        mk_stats = getattr(pipe, "last_mk_stats", {})

        results.append({
            "text":      text,
            "ttfc_ms":   ttfc_ms,
            "rtf":       rtf,
            "audio_s":   audio_dur_s,
            "total_ms":  total_t * 1000,
            "tok_per_s": mk_stats.get("tok_per_s", 0),
            "fallback":  mk_stats.get("fallback", False),
        })

    return results


# ── Rich output helpers ───────────────────────────────────────────────────────

def _target(value: float, target: float, lower_better: bool = True) -> str:
    """Return '[green]PASS[/green]' or '[red]FAIL[/red]' vs target."""
    ok = (value < target) if lower_better else (value > target)
    return f"[green]✓[/green]" if ok else f"[red]✗[/red]"


def print_decode_table(results: list[dict]) -> None:
    t = Table(title="Decode Throughput", show_header=True, header_style="bold cyan")
    t.add_column("Backend", style="dim", width=42)
    t.add_column("tok/s", justify="right")
    t.add_column("ms/tok", justify="right")
    t.add_column("vs target (1000 tok/s)", justify="center")

    for r in results:
        if "error" in r:
            t.add_row(r.get("backend", "?"), "ERROR", "ERROR", "[red]✗[/red]")
        else:
            t.add_row(
                r["backend"],
                f"{r['tok_per_s']:.0f}",
                f"{r['ms_per_tok']:.2f}",
                _target(r["tok_per_s"], 1000, lower_better=False),
            )
    console.print(t)


def print_tts_table(results: list[dict]) -> None:
    t = Table(title="TTS Latency (TTFC + RTF)", show_header=True, header_style="bold cyan")
    t.add_column("Text", width=38)
    t.add_column("TTFC (ms)", justify="right")
    t.add_column("<60ms?", justify="center")
    t.add_column("RTF", justify="right")
    t.add_column("<0.15?", justify="center")
    t.add_column("Audio", justify="right")
    t.add_column("tok/s", justify="right")

    for r in results:
        fallback_note = " [fallback]" if r.get("fallback") else ""
        t.add_row(
            r["text"][:37] + fallback_note,
            f"{r['ttfc_ms']:.0f}",
            _target(r["ttfc_ms"], 60),
            f"{r['rtf']:.3f}",
            _target(r["rtf"], 0.15),
            f"{r['audio_s']:.2f}s",
            f"{r['tok_per_s']:.0f}",
        )
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    all_results = {}

    console.rule("[bold blue]e3-assessment Benchmark Suite[/bold blue]")
    console.print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print()

    # ── Decode throughput ─────────────────────────────────────────────────────
    decode_results = []

    mk_result = bench_megakernel_talker(
        n_warmup=0 if args.quick else 2,
        n_runs=3 if args.quick else 5,
        max_tokens=50 if args.quick else 100,
    )
    decode_results.append(mk_result)
    all_results["megakernel_talker"] = mk_result

    if not args.no_megakernel:
        hf_result = bench_hf_baseline(
            n_warmup=0 if args.quick else 1,
            n_runs=2 if args.quick else 3,
            max_tokens=50 if args.quick else 100,
        )
        decode_results.append(hf_result)
        all_results["hf_baseline"] = hf_result

    print_decode_table(decode_results)

    # ── Full TTS pipeline ─────────────────────────────────────────────────────
    utterances = SHORT_UTTERANCES if args.quick else (SHORT_UTTERANCES + LONG_UTTERANCES)
    tts_results = await bench_full_tts(utterances)
    all_results["tts_latency"] = tts_results
    print_tts_table(tts_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_ttfc = [r["ttfc_ms"] for r in tts_results]
    valid_rtf  = [r["rtf"]     for r in tts_results if r["rtf"] > 0]
    mk_toks    = mk_result.get("tok_per_s", 0)

    console.rule("[bold green]Summary[/bold green]")
    console.print(f"  Megakernel talker:  [bold]{mk_toks:.0f}[/bold] tok/s")
    if valid_ttfc:
        console.print(f"  TTFC avg:           [bold]{sum(valid_ttfc)/len(valid_ttfc):.0f}[/bold] ms  (target <60 ms)")
    if valid_rtf:
        console.print(f"  RTF  avg:           [bold]{sum(valid_rtf)/len(valid_rtf):.3f}[/bold]  (target <0.15)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n  Results saved to [dim]{out_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="e3-assessment benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Fewer runs, faster")
    parser.add_argument("--no-megakernel", action="store_true", help="Skip HF baseline (saves time)")
    args = parser.parse_args()

    asyncio.run(main(args))
