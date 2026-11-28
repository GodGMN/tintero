# Tintero documentation

## Build and run

- [RX 7900 XTX quick start](../README.md#quick-start-on-rx-7900-xtx)
- [Build the HIP engine and tests](HIP-PORT.md#build-on-prometheus-without-a-gpu)
- [Export the tokenizer and start the API server](HIP-PORT.md#serve-the-iq4_nl-model-on-gfx1100)
- [Upstream RTX 3090 build and model files](INKLING2-MINI.md)

## Settings

- [RX 7900 XTX preset](inkling2-mini/ik2-mini-gfx1100.json)
- [Context size and FP4 cache profiles](HIP-PORT.md#named-fp4-profiles)
- [Concurrent requests and batch size](HIP-PORT.md#concurrent-decode-for-short-chat-requests)
- [Runtime ablation](inkling2-mini/runtime-ablation-gfx1100.md)
- [Sparse long-context decode](inkling2-mini/sparse-decode.md)

## Tests and measurements

- [Run or schedule the local GPU smoke check](GPU-SMOKE.md)
- [Run the kernel and host reference tests](HIP-PORT.md#verification)
- [Compare decode logits with the CPU reference](HIP-PORT.md#forced-decode-comparison)
- [RX 7900 XTX throughput and VRAM measurements](HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens)
