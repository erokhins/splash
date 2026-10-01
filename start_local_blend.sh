#!/bin/bash
# Start Splash on a local MLX model directory and its local DFlash2 draft,
# served under the model directory's name. Everything Splash writes (the
# installed assembly, prepared weights, runtime state) goes under
# SPLASH_DATA; the model and draft directories are only read. The first start
# installs the model (one read of its files) and prepares the weights (about
# the model's size); later starts reuse both.
#
# Usage: ./start_local_blend.sh [options passed to splash serve]
#   Overrides: SPLASH_TARGET, SPLASH_DRAFT, SPLASH_DATA.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SPLASH_TARGET:-$HOME/.local/share/junie-local/models/Qwen3.8-3.6-27B-blend-MLX-4bit}"
DRAFT="${SPLASH_DRAFT:-$HOME/.local/share/junie-local/models/Qwen3.8-3.6-27B-blend-DFlash2-MLX-4bit}"
DATA="${SPLASH_DATA:-$HOME/.local/share/junie-local/splash-cache}"

for path in "$TARGET/config.json" "$DRAFT/config.json"; do
    if [ ! -e "$path" ]; then
        echo "Missing model path: $path" >&2
        exit 1
    fi
done

echo "Starting Splash server with model: $(basename "$TARGET")"
echo "Target: $TARGET"
echo "Draft:  $DRAFT"
echo "Data:   $DATA"
echo ""

# server.py reads it; the launcher has no flag for it.
export SPLASH_PREFILL_ACTIVATIONS="${SPLASH_PREFILL_ACTIVATIONS:-int8}"

exec "$SCRIPT_DIR/splash" serve \
    --model "$TARGET" \
    --draft-model "$DRAFT" \
    --data-dir "$DATA" \
    "$@"
