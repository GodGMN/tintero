#!/bin/sh
# Stage the Python tools of the gfx1100 service image: usage: stage-ik2-tools.sh REPO_ROOT CONTEXT
#
# tools/ik2_ablate/ is the ablation lab (fitting, grading and benchmarking scripts, machine-specific wrappers). Only
# its directions/ folder reaches an image, and that goes to ablate/ in package-ik2-gfx1100.sh, so the lab itself is
# left out here. tools/ik2_ablate/MANIFEST.json lists which files are runtime assets.
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 REPO_ROOT CONTEXT" >&2
    exit 2
fi
root=$1
context=$2
mkdir -p "$context/tools"
for entry in "$root/tools"/ik2*; do
    [ -e "$entry" ] || continue
    [ "$(basename "$entry")" = ik2_ablate ] && continue
    cp -R "$entry" "$context/tools/"
done
for name in strata_tokenizer gguf_reader gguf_writer maya_quants maya_names ablate_fit needle_bench inkling2_gguf; do
    cp "$root/tools/$name.py" "$context/tools/"
done
