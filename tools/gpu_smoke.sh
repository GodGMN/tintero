#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--main] {hip|cuda} MODEL.gguf STATE_DIR" >&2
    exit 2
}

from_main=false
if [[ ${1:-} == --main ]]; then
    from_main=true
    shift
fi
[[ $# == 3 ]] || usage
backend=$1
[[ $backend == hip || $backend == cuda ]] || usage
model=$(realpath -e -- "$2")
[[ -f $model && -r $model ]] || usage
umask 077
mkdir -p -- "$3"
state=$(realpath -e -- "$3")

exec 9>>"$HOME/.tintero-gpu.lock"
if ! flock -n 9; then
    echo "Skipped: the GPU lock is held"
    exit 75
fi

skip() {
    echo "Skipped: $*"
    exit 75
}

gpu_idle() {
    local busy_message=${1:-Skipped}
    if [[ $backend == hip ]]; then
        local card busy used total found=0
        for card in /sys/class/drm/card[0-9]*/device; do
            [[ -r $card/mem_info_vram_total ]] || continue
            read -r total < "$card/mem_info_vram_total"
            read -r used < "$card/mem_info_vram_used"
            read -r busy < "$card/gpu_busy_percent"
            echo "AMD $(basename "$(realpath "$card")"): ${used}/${total} bytes VRAM, ${busy}% busy"
            if (( used > 1024**3 || busy != 0 )); then
                echo "$busy_message: AMD GPU in use"
                return 75
            fi
            found=1
        done
        if (( ! found )); then
            skip "AMD GPU counters unavailable"
        fi
    else
        local devices uuid name used total busy processes available_kib
        if ! devices=$(nvidia-smi --query-gpu=uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits); then
            skip "NVIDIA GPU counters unavailable"
        fi
        while IFS=, read -r uuid name used total busy; do
            echo "NVIDIA GPU: ${used}/${total} MiB VRAM, ${busy}% busy"
            if [[ ${name# } == "NVIDIA GB10" && ${used# } == "[N/A]" && ${total# } == "[N/A]" ]]; then
                if ! processes=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits); then
                    skip "NVIDIA compute process list unavailable"
                fi
                available_kib=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)
                if [[ -n $processes || ! $available_kib =~ ^[0-9]+$ ]] || (( available_kib < 32 * 1024**2 )); then
                    echo "$busy_message: GB10 has compute clients or insufficient available unified memory"
                    return 75
                fi
                used=0
            fi
            if [[ ! $used =~ ^[[:space:]]*[0-9]+$ || ! $busy =~ ^[[:space:]]*[0-9]+$ ]]; then
                skip "NVIDIA GPU counters unavailable"
            fi
            if (( used > 1024 || busy != 0 )); then
                echo "$busy_message: NVIDIA GPU in use"
                return 75
            fi
        done <<< "$devices"
    fi
}

today=$(date +%F)
if $from_main && [[ -f $state/last-main-attempt && $(<"$state/last-main-attempt") == "$today" ]]; then
    echo "The default-branch check already finished today"
    exit 0
fi

echo "Attempt: $(date -u +%FT%TZ) on $(hostname), $backend"
df -h -- "$state"
free_kib=$(df -Pk -- "$state" | awk 'END {print $4}')
if (( free_kib < 2 * 1024**2 )); then
    skip "less than 2 GiB free in $state"
fi
gpu_idle

scratch=$(mktemp -d "$state/run.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
src=$scratch/src
build=$scratch/build
mkdir -- "$src" "$build"

run_smoke() {
    local revision
    if $from_main; then
        git clone --depth 1 --single-branch --branch main https://github.com/GodGMN/tintero.git "$src"
        revision=$(git -C "$src" rev-parse HEAD)
    else
        local checkout
        checkout=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
        revision=$(git -C "$checkout" rev-parse HEAD)
        git -C "$checkout" archive "$revision" | tar -x -C "$src"
    fi
    echo "Revision: $revision"
    printf '%s  %s\n' cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4 "$model" |
        sha256sum --check

    local -a runtime=() devices=() cmake_args
    if [[ $backend == hip ]]; then
        cmake_args=(-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100
                    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm)
        if [[ -n ${TINTERO_HIP_IMAGE:-} ]]; then
            runtime=(docker run --pull=never --rm --network none --user "$(id -u):$(id -g)"
                     --mount "type=bind,src=$src,dst=$src,readonly"
                     --mount "type=bind,src=$build,dst=$build"
                     --entrypoint /usr/bin/env)
            devices=(--device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)"
                     --mount "type=bind,src=$model,dst=$model,readonly")
        fi
    else
        cmake_args=(-DSTRATA_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${TINTERO_CUDA_ARCH:-native}")
        nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
    fi

    local -a build_env=() gpu_env=()
    if (( ${#runtime[@]} )); then
        build_env=("${runtime[@]}" "$TINTERO_HIP_IMAGE")
        gpu_env=("${runtime[@]}" "${devices[@]}" "$TINTERO_HIP_IMAGE")
        docker image inspect --format 'ROCm image: {{.Id}}' "$TINTERO_HIP_IMAGE"
    fi
    if [[ $backend == hip ]]; then
        "${build_env[@]}" cat /opt/rocm/.info/version
    fi

    set -x
    "${build_env[@]}" cmake --version
    "${build_env[@]}" cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON "${cmake_args[@]}"
    "${build_env[@]}" nice cmake --build "$build" --target tintero ik2_kernels_test --parallel 2
    { set +x; } 2>/dev/null
    gpu_idle

    local flag
    for flag in "${!IK2_@}"; do
        unset "$flag"
    done
    set -x
    "${gpu_env[@]}" timeout --kill-after=10s 300s "$build/ik2_kernels_test"
    { set +x; } 2>/dev/null
    local attempt
    for attempt in {1..6}; do
        sleep 2
        if (gpu_idle "Waiting for idle GPU after parity"); then
            break
        fi
        if (( attempt == 6 )); then
            skip "GPU did not become idle after parity"
        fi
    done

    echo 'Prompt: The capital of France is'
    local output=$scratch/decode.txt expected=152717,13
    set -x
    "${gpu_env[@]}" env XDG_CACHE_HOME="$build/cache" IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 \
        IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1 timeout --kill-after=10s 300s "$build/tintero" \
        --model "$model" --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 2 |
        tee "$output"
    { set +x; } 2>/dev/null
    if [[ $(grep -Ex '[0-9]+(,[0-9]+)*' "$output") != "$expected" ]]; then
        echo "FAIL: expected greedy token IDs $expected" >&2
        return 1
    fi
    echo "PASS: kernel parity and greedy token IDs $expected"
}

set +e
(set -e; run_smoke) 2>&1 | tee "$state/last-run.log"
status=$?
set -e
if $from_main && (( status != 75 )); then
    printf '%s\n' "$today" > "$state/last-main-attempt"
fi
exit "$status"
