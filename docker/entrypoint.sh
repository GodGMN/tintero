#!/bin/sh
set -eu

for arg in "$@"; do
    case "$arg" in
        -h|--help) exec python3 serve/server.py --help ;;
    esac
done

config=${TINTERO_CONFIG:?set TINTERO_CONFIG to a preset JSON path}
config=$(python3 docker/prepare-config.py "$config" "$@")
exec python3 serve/server.py --engine maya \
    --host "${TINTERO_HOST:-0.0.0.0}" --port "${TINTERO_PORT:-8080}" "$@" --config "$config"
