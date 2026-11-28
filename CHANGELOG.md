# Changelog

Tintero releases are published on GitHub (Releases). Changes that are merged but not yet in a release are listed under
Unreleased. `VERSION` holds the number of the latest release.

## Unreleased

- **Fix: About > Updates in installed copies.** A Docker image or release tarball has no `.git`, so the update check
  fell back to a different project's releases, with that project's version number and user agent. It now asks
  `GodGMN/tintero` with the user agent `tintero/<version>`, and `VERSION` is `0.1.0`, the number of the published
  release. A git checkout still asks its own `origin`, so a fork checks its own releases.
- **`TINTERO_UPDATE_CHECK=0` turns the update check off**, as `MAYA_UPDATE_CHECK=0` does. If both are set,
  `TINTERO_UPDATE_CHECK` decides.

## v0.1.0 - 2026-11-27

First public release.

- **Images and tarballs:** CUDA (`sm_80` to `sm_121a`, Linux x86_64 and arm64) and AMD HIP (`gfx1100`, Linux x86_64)
  Docker images on GHCR, and Ubuntu 24.04 tarballs with `SHA256SUMS.txt`.
- **NVIDIA CUDA:** full-resident inference, bit-exact continuous batching (`IK2_BATCH`), grouped Tensor-Core MoE
  (`IK2_BATCH_GROUP_MOE=1/2`), native Blackwell block-scaled FP4 MMA (`IK2_BLACKWELL_NVFP4=1`), `IK2_KV=fp4-global`
  long-context KV, graphed MTP speculative decoding (`--draft`) and pinned-staging fast model loads.
- **AMD HIP (`gfx1100`, ROCm 7.2):** full-resident RDNA3 inference, FP8 and FP4-global KV, and graphed MTP speculative
  decoding.
- **Presets** in `docs/inkling2-mini/`: `ik2-mini-gfx1100.json`, `cuda-24gb-fp4.json`, `cuda-80gb.json`,
  `dgx-spark-quality.json`, `dgx-spark-batched-nvfp4.json` and `dgx-spark-longctx-fp4.json`.
- **Runtime abliteration** is optional: `IK2_ABLATE_MODE=on`.
