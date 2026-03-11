"""
megakernel package — adapts AlpinDale's qwen_megakernel for Qwen3-TTS talker.

The original kernel targets Qwen3-0.6B (28 layers, hidden=1024).
Qwen3-TTS talker uses the same hidden/head sizes but 20 layers.
We reuse the kernel binary verbatim and pass num_layers=20 at call time
— the layer loop is runtime-parameterized, so no recompilation is needed.
"""
