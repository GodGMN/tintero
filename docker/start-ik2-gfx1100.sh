#!/bin/sh
set -eu

preset=ik2-mini-gfx1100.json
if [ "${IK2_TEXT_ONLY:-0}" = 1 ] && [ -f ik2-mini-gfx1100-text.json ]; then
    preset=ik2-mini-gfx1100-text.json
elif [ -f ik2-mini-gfx1100-hip-host-vision.json ]; then
    case "${IK2_CACHE_SLOTS:-1}" in
        1|2) preset=ik2-mini-gfx1100-hip-host-vision.json ;;
        3|4) preset=ik2-mini-gfx1100-hip-host-vision-512.json ;;
    esac
elif [ "${IK2_CACHE_SLOTS:-1}" = 1 ] && [ -f ik2-mini-gfx1100-hip-vision.json ]; then
    preset=ik2-mini-gfx1100-hip-vision.json
fi

case "${IK2_ABLATE_MODE:-off}" in
    ""|0|off|false|none) ;;
    1|on|true|cyber|v21-pc4-kn)
        if [ -z "${IK2_ABLATE:-}" ]; then
            if [ -r /opt/maya/models/v21-pc4.f32 ]; then
                IK2_ABLATE=/opt/maya/models/v21-pc4.f32
            else
                IK2_ABLATE=/opt/maya/ablate/v21-pc4.f32
            fi
        fi
        export IK2_ABLATE
        export IK2_ABLATE_SCALE="${IK2_ABLATE_SCALE:-1.6}"
        export IK2_ABLATE_VISITS="${IK2_ABLATE_VISITS:-22:48}"
        export IK2_ABLATE_KEEP_NORM="${IK2_ABLATE_KEEP_NORM:-1}"
        ;;
    *)
        echo "Unknown IK2_ABLATE_MODE=${IK2_ABLATE_MODE} (expected off|cyber|v21-pc4-kn)" >&2
        exit 1
        ;;
esac

preset_override=${TINTERO_IK2_PRESET-${MAYA_IK2_PRESET-}}
exec python3 serve/server.py --engine maya --config "${preset_override:-$preset}" "$@"
