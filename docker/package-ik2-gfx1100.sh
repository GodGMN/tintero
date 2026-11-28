#!/bin/sh
# Package a built gfx1100 engine and a tokenizer exported from the target GGUF.
set -eu
if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
    echo "usage: $0 TINTERO_BINARY TOKENIZER_DIRECTORY IMAGE_TAG [CPU_VISION [HIP_VISION [HIP_VISION_HOST_WEIGHTS]]]" >&2
    exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test -x "$1"
test -r "$2/chat_template.jinja"
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$root/LICENSE" "$root/VERSION" "$context/"
cp "$1" "$context/tintero"
cp -R "$2" "$context/tokenizer"
cp "$root/docs/inkling2-mini/ik2-mini-gfx1100.json" "$root/docker/ik2-gfx1100.Dockerfile" "$context/"
cp "$root/docker/start-ik2-gfx1100.sh" "$context/"
mkdir "$context/ablate"
if [ -d "$root/tools/ik2_ablate/directions" ]; then
    cp -R "$root/tools/ik2_ablate/directions/." "$context/ablate/"
fi
mkdir "$context/vision"
if [ "$#" -ge 4 ]; then
    test -x "$4"
    cp "$4" "$context/vision/maya-vision-cpu"
    if [ "$#" -ge 5 ]; then
        test -x "$5"
        cp "$5" "$context/vision/maya-vision-hip"
    fi
    if [ "$#" -eq 6 ]; then
        test -x "$6"
        cp "$6" "$context/vision/maya-vision-hip-host"
    fi
    python3 - "$context" <<'PYCONFIG'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
default = directory / "ik2-mini-gfx1100.json"
cfg = json.loads(default.read_text())
(directory / "ik2-mini-gfx1100-text.json").write_text(default.read_text())
cfg["vision"] = dict(exe="vision/maya-vision-cpu", model=cfg["args"][1], mmproj="models/mmproj-F16.gguf",
                     threads=4, max_tokens=1024, gpu=False, resident=True)
default.write_text(json.dumps(cfg, indent=2) + "\n")

def gpu_preset(exe, name, chunk):
    if (directory / exe).is_file():
        cfg["vision"].update(exe=exe, gpu=True)
        cfg["args"][cfg["args"].index("--chunk") + 1] = str(chunk)
        (directory / name).write_text(json.dumps(cfg, indent=2) + "\n")

gpu_preset("vision/maya-vision-hip", "ik2-mini-gfx1100-hip-vision.json", 256)
gpu_preset("vision/maya-vision-hip-host", "ik2-mini-gfx1100-hip-host-vision.json", 1024)
gpu_preset("vision/maya-vision-hip-host", "ik2-mini-gfx1100-hip-host-vision-512.json", 512)
host_vision = directory / "ik2-mini-gfx1100-hip-host-vision-512.json"
if host_vision.is_file():
    for slots, per_slot in [(2, 360448), (4, 176128)]:
        preset = json.loads(host_vision.read_text())
        preset["env"]["IK2_KV"] = "fp4-global"
        args = preset["args"]
        args[args.index("--max-context") + 1] = str(slots * per_slot)
        args.extend(["--cache-slots", str(slots)])
        (directory / f"ik2-mini-gfx1100-fp4-{slots}.json").write_text(json.dumps(preset, indent=2) + "\n")
PYCONFIG
fi
cp -R "$root/serve" "$context/"
mkdir "$context/tests"
cp "$root/tests"/ik2*.py "$context/tests/"
cp -R "$root/ref" "$context/ref"
sh "$root/docker/stage-ik2-tools.sh" "$root" "$context"
cp "$root/third_party/ggml/LICENSE" "$context/LICENSE.ggml"
cp "$root/third_party/ciru/LICENSE" "$context/LICENSE.ciru"
cp "$root/third_party/ciru/NOTICE" "$context/NOTICE.ciru"
find "$context" -name __pycache__ -type d -prune -exec rm -rf '{}' +
docker buildx build --load --resource memory=6g --resource memory-swap=6g \
    --resource cpu-period=100000 --resource cpu-quota=400000 \
    -t "$3" -f "$context/ik2-gfx1100.Dockerfile" "$context"
