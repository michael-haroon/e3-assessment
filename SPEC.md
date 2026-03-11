# e3-assessment — Project Specification

## TL;DR
Wire AlpinDale's `qwen_megakernel` (Qwen3-0.6B at ~1,000 tok/s on RTX 5090) into Qwen3-TTS's talker decoder stage, then serve it inside a Pipecat voice pipeline.

---

## Hardware
- **GPU:** RTX 5090 (sm_120 / Blackwell, GDDR7)
- **Host:** Vast.ai rental (compute costs reimbursed)
- **Requirement:** CUDA 12.8+, bfloat16 only — no quantization

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Megakernel tok/s | ≥ 1,000 tok/s | Reference: 1,036 tok/s on RTX 5090 |
| TTFC | < 60 ms | Time to first audio chunk |
| RTF | < 0.15 | 1s audio generated in < 300ms wall time |
| Streaming | Frame-by-frame | Do NOT buffer full utterance before sending |

These are reference benchmarks, not hard pass/fail cutoffs — but document any gaps honestly.

---

## What the Megakernel Does

- **Architecture:** 128 persistent thread blocks × 512 threads, single non-cooperative CUDA kernel
- **Model:** Qwen3-0.6B in bfloat16 (no quantization)
- **Performance:** ~1,000 tok/s decode (0.97 ms/step), 71% theoretical GDDR7 bandwidth
- **Output:** Single-token argmax per step (autoregressive decode loop on host)
- **Source:** https://github.com/AlpinDale/qwen_megakernel
- **Blog:** https://blog.alpindale.net/posts/5090_decode_optimization/

---

## Architecture

```
Microphone / STT
      │
      ▼
   Pipecat Pipeline
      │
      ├─ STT (Deepgram / Whisper)
      │
      ├─ LLM (response text)
      │
      └─ TTS ← THIS PROJECT
           │
           ▼
    Qwen3-TTS-12Hz-0.6B-CustomVoice
           │
    ┌──────┴────────────────────────────┐
    │  PyTorch Prefill (HF generate=1)  │  ~40ms warm
    │  Speaker embed + text → 1st token │
    └──────┬────────────────────────────┘
           │
    ┌──────┴────────────────────────────┐
    │  Megakernel Decode (~1ms/token)   │  ~600ms for 600 tokens
    │  CUDA kernel, bfloat16, sm_120    │
    └──────┬────────────────────────────┘
           │
    ┌──────┴────────────────────────────┐
    │  CosyVoice Vocoder (1 call)       │  ~300ms
    │  Full token sequence → PCM        │
    └──────┬────────────────────────────┘
           │  int16 PCM chunks, 24 kHz mono
           ▼
    Pipecat audio output → speaker
```

---

## Steps

### Step 1 — Adapt Megakernel for Qwen3-TTS
- Clone https://github.com/AlpinDale/qwen_megakernel
- Qwen3-TTS talker backbone is architecturally identical to Qwen3-0.6B but has 20 layers (not 28)
- Pass `num_layers=20` at decode time — no kernel recompile needed
- Weight layout differences documented in `megakernel/tts_talker_decoder.py`

### Step 2 — Build Inference Server
- `tts/qwen3_tts_pipeline.py` — streaming TTS pipeline
- Monkey-patches `inner_lm.generate` inside `generate_custom_voice()` to replace the HF autoregressive loop with the megakernel
- Returns audio as `asyncio.Queue`-fed chunks to caller

### Step 3 — Integrate with Pipecat
- Wire into `pipeline/run_bot.py`
- STT → LLM → `Qwen3TTSTTSService` → audio output
- Pipecat service interface: push int16 PCM frames as they're ready

### Step 4 — Validate End-to-End
```bash
cd e3-assessment
bash setup.sh
python validate.py --skip-tts   # fast kernel sanity check
python validate.py               # full TTS test (TTFC + RTF)
python pipeline/run_bot.py       # open printed URL in browser
python benchmarks/benchmark.py   # performance numbers
```

---

## Realistic Performance Numbers (RTX 5090, warm model)

| Component | Time |
|-----------|------|
| Model load (once) | ~2.5s |
| Warm-up synthesis | ~1.2s |
| Prefill (warm) | ~40ms |
| Megakernel 600 tokens | ~600ms |
| Vocoder (1 call) | ~300ms |
| **TTFC (warm)** | **~940ms** |
| **RTF** | **~0.15** |
| Megakernel throughput | ~1,000 tok/s |

### Why TTFC > 60ms
The 60ms target requires the vocoder to start producing audio before the full token sequence is generated (incremental vocoding). CosyVoice's flow-matching decoder conditions on the full token sequence globally — calling it on partial prefixes causes hangs on longer sequences. The safe architecture is one vocoder call on the complete output, giving TTFC = prefill + decode + vocoder (~940ms warm).

True sub-60ms TTFC would require either:
1. A streaming-capable vocoder (e.g. Griffin-based or chunk-causal flow matching)
2. KV cache pre-warming before text arrives (speculative prefill)
3. A different TTS backend with a causal vocoder

---

## Key Files

| File | Purpose |
|------|---------|
| `megakernel/kernel.cu` | 1,200-line CUDA megakernel (sm_120, bf16) |
| `megakernel/tts_talker_decoder.py` | Stateful decoder wrapping the kernel |
| `megakernel/tts_talker_build.py` | JIT compile + load extension |
| `tts/qwen3_tts_pipeline.py` | End-to-end TTS pipeline with streaming |
| `pipeline/run_bot.py` | Pipecat voice agent |
| `validate.py` | End-to-end validation + benchmarks |
| `benchmarks/benchmark.py` | Standalone performance numbers |

---

## References
- Megakernel source: https://github.com/AlpinDale/qwen_megakernel
- Blog post: https://blog.alpindale.net/posts/5090_decode_optimization/
- Pipecat docs: https://docs.pipecat.ai
- Qwen3-TTS model: https://huggingface.co/Qwen/Qwen3-TTS
- Qwen3-TTS-CustomVoice: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
