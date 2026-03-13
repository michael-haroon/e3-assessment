"""
parity_test.py
──────────────
Deterministic parity check between:
  1) Standard PyTorch talker generation path
  2) Megakernel-backed talker generation path

Expectation:
  Token IDs must match step-by-step for the configured number of steps.

Usage:
  python benchmarks/parity_test.py
  python benchmarks/parity_test.py --prompt "Hello" --steps 10 --seed 1234
  python benchmarks/parity_test.py --keep-vocoder   # if you explicitly want full audio decode
"""

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bootstrap  # noqa: F401
from tts.qwen3_tts_pipeline import Qwen3TTSPipeline


@dataclass
class ParityResult:
    prompt: str
    steps: int
    seed: int
    baseline_tokens: list[int]
    megakernel_tokens: list[int]

    @property
    def matched(self) -> bool:
        return self.baseline_tokens == self.megakernel_tokens


@dataclass
class RunMeta:
    skip_vocoder: bool


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _optional_noop_vocoder(pipe: Qwen3TTSPipeline, skip_vocoder: bool):
    """
    For token-parity we don't need waveform decoding.
    CosyVoice decode can be memory-heavy and may destabilize small instances.
    """
    if not skip_vocoder:
        yield
        return

    tokenizer = pipe._qwen_model.model.speech_tokenizer
    original_decode = tokenizer.decode

    def _fake_decode(items):
        # Keep output shape compatible with callers while doing near-zero work.
        n = len(items)
        wavs = [np.zeros(1, dtype=np.float32) for _ in range(n)]
        return wavs, 24000

    tokenizer.decode = _fake_decode
    try:
        yield
    finally:
        tokenizer.decode = original_decode


def _run_baseline(pipe: Qwen3TTSPipeline, prompt: str, steps: int, skip_vocoder: bool) -> list[int]:
    inner_lm = pipe._qwen_model.model.talker
    original_generate = inner_lm.generate
    captured = {"tokens": None}

    def _capture_generate(*args, **kwargs):
        out = original_generate(*args, **kwargs)
        captured["tokens"] = [int(x) for x in out.sequences[0].detach().cpu().tolist()]
        return out

    inner_lm.generate = _capture_generate
    try:
        with _optional_noop_vocoder(pipe, skip_vocoder):
            pipe._qwen_model.generate_custom_voice(
                text=prompt,
                language=pipe.language,
                speaker=pipe.speaker,
                max_new_tokens=steps,
                do_sample=False,
                repetition_penalty=1.0,
                subtalker_dosample=False,
                subtalker_top_p=1.0,
                subtalker_temperature=1.0,
            )
    finally:
        inner_lm.generate = original_generate

    if not captured["tokens"]:
        raise RuntimeError("Baseline run did not capture talker tokens")

    return captured["tokens"][:steps]


def _run_megakernel(pipe: Qwen3TTSPipeline, prompt: str, steps: int, skip_vocoder: bool) -> list[int]:
    pipe.max_new_tokens = steps
    pipe._generation_kwargs = {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "subtalker_dosample": False,
        "subtalker_top_p": 1.0,
        "subtalker_temperature": 1.0,
    }

    with _optional_noop_vocoder(pipe, skip_vocoder):
        pipe._synthesize_with_megakernel(prompt)

    tokens = pipe.last_mk_stats.get("generated_tokens", [])
    if not tokens:
        raise RuntimeError("Megakernel run did not capture generated token trace")
    return [int(x) for x in tokens[:steps]]


def run_parity(prompt: str, steps: int, seed: int, skip_vocoder: bool) -> tuple[ParityResult, RunMeta]:
    _set_seed(seed)

    pipe = Qwen3TTSPipeline(max_new_tokens=steps, verbose=False)
    pipe._ensure_loaded()

    with torch.inference_mode():
        baseline = _run_baseline(pipe, prompt, steps, skip_vocoder=skip_vocoder)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _set_seed(seed)

    with torch.inference_mode():
        megakernel = _run_megakernel(pipe, prompt, steps, skip_vocoder=skip_vocoder)

    return (
        ParityResult(
            prompt=prompt,
            steps=steps,
            seed=seed,
            baseline_tokens=baseline,
            megakernel_tokens=megakernel,
        ),
        RunMeta(skip_vocoder=skip_vocoder),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Megakernel parity test")
    parser.add_argument("--prompt", default="Hello", help="Prompt text for parity run")
    parser.add_argument("--steps", type=int, default=10, help="Number of decode steps to compare")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--keep-vocoder",
        action="store_true",
        help="Run full vocoder decode too (heavier; unnecessary for token parity)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("benchmarks", "parity_results.json"),
        help="JSON output path",
    )
    args = parser.parse_args()

    result, meta = run_parity(
        prompt=args.prompt,
        steps=args.steps,
        seed=args.seed,
        skip_vocoder=not args.keep_vocoder,
    )

    payload = {
        "prompt": result.prompt,
        "steps": result.steps,
        "seed": result.seed,
        "matched": result.matched,
        "baseline_tokens": result.baseline_tokens,
        "megakernel_tokens": result.megakernel_tokens,
        "skip_vocoder": meta.skip_vocoder,
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Baseline  :", result.baseline_tokens)
    print("Megakernel:", result.megakernel_tokens)
    print("skip_vocoder:", meta.skip_vocoder)
    if not result.matched:
        for i, (a, b) in enumerate(zip(result.baseline_tokens, result.megakernel_tokens), start=1):
            if a != b:
                print(f"Mismatch at step {i}: baseline={a} megakernel={b}")
                break
        if len(result.baseline_tokens) != len(result.megakernel_tokens):
            print(
                f"Length mismatch: baseline={len(result.baseline_tokens)} "
                f"megakernel={len(result.megakernel_tokens)}"
            )
        return 1

    print("PASS: token parity matched for all compared steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
