# Local GPU smoke check

Use a Linux GPU host with a stock ggml-org Inkling 2 Mini IQ4_NL GGUF and enough free VRAM for the full model. The smoke check builds the engine and runs `ik2_kernels_test` against its reference calculations. For the prompt “The capital of France is”, it expects the two greedy token IDs for “ Paris.” (`152717,13`).

```bash
tools/gpu_smoke.sh hip /path/to/Inkling-2-Mini-IQ4_NL.gguf ~/.local/state/tintero-nightly
tools/gpu_smoke.sh cuda /path/to/Inkling-2-Mini-IQ4_NL.gguf ~/.local/state/tintero-nightly
```

The manual check uses the current committed revision. HIP builds for gfx1100 with ROCm in `/opt/rocm`. To use ROCm in Docker, set `TINTERO_HIP_IMAGE` to a locally installed image with ROCm, CMake, Make, a C++ compiler and `timeout`. The script runs as your UID, with read-only mounts for the source and model. CUDA uses the native toolkit and detects the device architecture by default. Set `TINTERO_CUDA_ARCH` for an explicit CMake architecture.

The command verifies the model SHA-256 and uses a temporary source and build directory. It records the revision, commands, GPU and results in `last-run.log` in the state directory. Exit code 75 means the lock, GPU or disk check deferred the run. Use `flock "$HOME/.tintero-gpu.lock" …` for other GPU jobs started by the same account so they share the reservation.

## Nightly schedule

Install a reviewed copy of the script outside any checkout. With `--main`, it fetches a fresh shallow copy of `main` from the canonical GitHub repository. Keep the scheduled command under the GPU host account.

```bash
install -d ~/.local/bin ~/.local/state/tintero-nightly
install -m 700 tools/gpu_smoke.sh ~/.local/bin/tintero-gpu-smoke
```

With user cron, run it every 15 minutes in an overnight window and use absolute paths. This example starts at 03:00 local time and retries deferred checks through 06:45. The first pass or failure stops retries for the day. Existing crontab entries can stay in place.

```cron
*/15 3-6 * * * /home/operator/.local/bin/tintero-gpu-smoke --main hip /path/to/Inkling-2-Mini-IQ4_NL.gguf /home/operator/.local/state/tintero-nightly > /home/operator/.local/state/tintero-nightly/last-poll.log 2>&1
```

Read `last-poll.log` for the latest scheduler decision and `last-run.log` for the build and checks. The check covers one short greedy decode and the backend's default kernel suite.
