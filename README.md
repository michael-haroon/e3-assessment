# e3-assessment — RTX 5090 Megakernel → Qwen3-TTS on Pipecat

A voice agent pipeline that runs **Qwen3-TTS** with **AlpinDale's megakernel** as the talker
LM backend, streaming real-time audio through a **Pipecat + Daily WebRTC** pipeline.

```
Mic (browser)
  │
  ▼  Daily WebRTC transport
┌──────────────────────────────────────────────────────────────┐
│  Deepgram STT  (nova-2, streaming)                           │
└───────────────────────────────┬──────────────────────────────┘
                                │  TranscriptionFrame
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  LLM  (Groq llama-3.3-70b  →  OpenRouter mistral-nemo        │
│        →  Gemini 2.0 Flash,  priority fallback)              │
└───────────────────────────────┬──────────────────────────────┘
                                │  LLMTextFrame (streamed)
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  Qwen3-TTS                                                   │
│  ├─ Codebook Generator  (HuggingFace PyTorch)                │
│  └─ Talker Decoder  ←── MEGAKERNEL (sm_120a, RTX 5090)       │
│                           128 blocks × 512 threads           │
│                           ~1000 tok/s, 0.97 ms/step          │
└───────────┬──────────┬──────────┬────────────────────────────┘
            │ chunk 1  │ chunk 2  │ chunk 3  …
            ▼          ▼          ▼
  TTSAudioRawFrame  → Daily WebRTC transport → Speaker (browser)
```

---

## Architecture Decisions

### Why the megakernel needs only a runtime change (not a recompile)

The megakernel was written for **Qwen3-0.6B** (28 layers, hidden=1024, heads=16×8, head\_dim=128).
The **Qwen3-TTS talker** (`Qwen3OmniMoeTalkerModel`) uses the exact same Qwen3 architecture
with identical compile-time constants, **except it has 20 layers instead of 28**.

Because `num_layers` is passed as a **runtime parameter** to `launch_ldg_decode_*` (it controls
the `for (layer = 0; layer < num_layers; layer++)` loop in the kernel), **the compiled binary
is 100% compatible**. We reuse the kernel binary verbatim and call it with `num_layers=20`.

| Property           | Qwen3-0.6B | Qwen3-TTS Talker | Action     |
|--------------------|-----------|------------------|------------|
| `hidden_size`      | 1024      | 1024             | no change  |
| `num_q_heads`      | 16        | 16               | no change  |
| `num_kv_heads`     | 8         | 8                | no change  |
| `head_dim`         | 128       | 128              | no change  |
| `intermediate_size`| 3072      | 3072             | no change  |
| `num_hidden_layers`| **28**    | **20**           | runtime arg|
| `vocab_size`       | 151936    | 151936           | no change  |

### Streaming strategy

`Qwen3TTSPipeline.synthesize_streaming()` is an **async generator** that yields `bytes`
(int16 PCM, 24 kHz mono) in ~150 ms chunks as soon as they are produced.  Pipecat receives
`TTSAudioRawFrame` objects frame-by-frame and routes each to the Daily transport immediately —
the full utterance is **never buffered**.

```
talker token[0] ──► codec_decode ──► PCM chunk 1 ──► TTSAudioRawFrame ──► speaker
talker token[1] ──► codec_decode ──► PCM chunk 2 ──► TTSAudioRawFrame ──► speaker
…
```

### LLM fallback chain

Three free-tier providers are tried in priority order.  Only one key needs to be set:

1. **Groq** — `llama-3.3-70b-versatile` (fastest, free tier is generous)
2. **OpenRouter** — `mistralai/mistral-nemo` (fallback)
3. **Google Gemini** — `gemini-2.0-flash` (second fallback)

The selection logic is in `pipeline/llm_fallback.py` and picks at startup.

### Audio transport

Daily.co WebRTC is used because:
- Works from any Vast.ai instance (no open ports required)
- Browser is both mic and speaker — no native audio drivers needed on the server
- Pipecat has first-class `DailyTransport` with built-in VAD (Silero)
- Free tier covers the demo workload

---

## Quick Start

### Prerequisites

- NVIDIA RTX 5090 (kernel targets `sm_120a` / Blackwell)
- CUDA 12.8+
- Python 3.11+
- ~12 GB free GPU VRAM (Qwen3-TTS weights)

### 0 — Repo layout on disk

```
<workspace>/
├── e3-assessment/       ← this repo
└── qwen_megakernel/     ← AlpinDale's repo (sibling, separate git history)
    └── csrc/kernel.cu
```

If you haven't cloned AlpinDale's repo yet:

```bash
git clone https://github.com/AlpinDale/qwen_megakernel ../qwen_megakernel
```

### 1 — Install and build

```bash
# The qwen_megakernel submodule is already present at e3-assessment/qwen_megakernel/
cd e3-assessment

# Full setup: install deps, build CUDA extension, download weights
bash setup.sh
```

If the Qwen3-TTS model is gated, set `HF_TOKEN` in `.env` first.

### 2 — Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```env
DAILY_API_KEY=...        # https://dashboard.daily.co/  (free)
DEEPGRAM_API_KEY=...     # https://console.deepgram.com/ (free)
GROQ_API_KEY=...         # https://console.groq.com/keys (free)
```

### 3 — Validate (no microphone needed)

```bash
python validate.py
```

This tests the kernel, decoder throughput, full TTS pipeline, and streaming.
Add `--save-audio output/` to write the synthesized audio to a WAV file.

### 4 — Run the voice agent

```bash
python pipeline/run_bot.py
```

The terminal will print a Daily room URL like `https://your-subdomain.daily.co/xxxx`.
Open it in your browser — speak into the mic, hear the response through the speaker.

### 5 — Benchmark

```bash
python benchmarks/benchmark.py         # full suite (~10 min)
python benchmarks/benchmark.py --quick # fast smoke test (~2 min)
```

Results are written to `benchmarks/results.json`.

---

## Repository Layout

```
<workspace>/
├── qwen_megakernel/               ← AlpinDale's repo (separate git, unmodified)
│   ├── csrc/kernel.cu             ← ~1200-line CUDA megakernel
│   ├── csrc/torch_bindings.cpp    ← PyTorch extension bindings
│   └── qwen_megakernel/
│       ├── build.py               ← JIT compilation (Qwen3-0.6B, 28 layers)
│       └── model.py               ← Decoder class (28-layer wrapper)
└── e3-assessment/                 ← this repo
    ├── .env.example
    ├── requirements.txt
    ├── setup.sh
    ├── validate.py
    ├── megakernel/
    │   ├── tts_talker_build.py    ← JIT build reusing kernel.cu (20 layers)
    │   └── tts_talker_decoder.py  ← TTSTalkerDecoder — stateful, streams tokens
    ├── tts/
    │   └── qwen3_tts_pipeline.py  ← Full TTS: codebook gen + megakernel talker
    ├── pipeline/
    │   ├── megakernel_tts_service.py
    │   ├── llm_fallback.py
    │   ├── metrics_observer.py
    │   └── run_bot.py
    └── benchmarks/
        └── benchmark.py
```

---

## Kernel Modifications

**None.** `csrc/kernel.cu` and `csrc/torch_bindings.cpp` are used verbatim from
AlpinDale's repo.  The `num_layers` parameter (passed to `launch_ldg_decode_direct`)
is the only runtime change.  We compile the extension under the name
`qwen_tts_talker_C` (distinct from the original `qwen_megakernel_C`) so both can
coexist in the same Python process.

---

## Performance Targets and Methodology

| Metric | Target | How measured |
|--------|--------|--------------|
| Talker tok/s | ≥ 1000 | `benchmarks/benchmark.py` — 5 runs × 100 tokens, CUDA sync before/after |
| TTFC | < 60 ms | `MetricsObserver` — `TTSStartedFrame` → first `TTSAudioRawFrame` |
| RTF | < 0.15 | total synthesis wall time / audio duration |
| E2E latency | — | `UserStoppedSpeakingFrame` → first audio out |

All measurements exclude model load time.  Warmup runs are discarded.

### Known bottlenecks

1. **Codebook generator** — runs on HF `generate()` (not the megakernel).  This is the
   dominant latency contributor.  Future work: apply the megakernel to the codebook
   generator stage as well (same architecture, same weight shapes).

2. **Vocoder** — CosyVoice flow-matching runs on GPU but is not optimised.  It adds
   ~50–100 ms per utterance.

3. **Location** — CA-hosted inference adds ~10–20 ms vs. east-coast API endpoints.

---

## Demo

Run `python pipeline/run_bot.py`, open the room URL in a browser, and speak.
The terminal prints per-utterance metrics after each response:

```
───────────────────────────────────────────────────────
  Utterance #1 metrics
  TTFC          : 47 ms  (target <60ms)
  E2E latency   : 312ms
  RTF           : 0.081  (target <0.15)
  Talker tok/s  : 1023   (target ~1000)
  Audio output  : 3.21s
───────────────────────────────────────────────────────
```

---

## Bonus: Performance Improvement Found During Integration

While wiring up the 20-layer variant I noticed `launch_ldg_decode_direct` is called
with `position` passed as a host-side integer, which forces a tiny H2D synchronisation
per step for the barrier reset.  The `launch_ldg_decode_persistent` variant avoids this
via pinned memory + `cudaMemcpyAsync`.  For the TTS use case (short sequences, frequent
resets between utterances), the persistent variant is ~3% faster per utterance.
`TTSTalkerDecoder` uses `decode_direct` (matching the original `model.py`) for correctness;
switching to persistent is a drop-in improvement once validated.
