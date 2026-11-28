#!/bin/sh
# Compatibility launcher for older update notices.
cd "$(dirname "$0")" || exit 1
exec ./maya.sh "$@"
