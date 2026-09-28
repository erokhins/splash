#!/bin/bash
# Start Splash server with the Qwen3.8-3.6-27B blend + DFlash2 v0.3 packed model.
# Mirrors the mlx-vlm gateway Splash adapter (mlx_vlm_gateway/splash.py).
# Usage: ./start_splash.sh [options passed to server.py]
#   Override the package location with SPLASH_PACKAGE=/path/to/package.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO="vvsotnikov/Qwen3.8-3.6-27B-blend-Splash-v0.3"
BINARY="$SCRIPT_DIR/build/splash"

# Find a suitable Python
PYTHON=""
for candidate in "$SCRIPT_DIR/.venv/bin/python" python3.13 python3 python3.12 python3.14; do
    if (test -x "$candidate" 2>/dev/null || command -v "$candidate" >/dev/null 2>&1) \
        && "$candidate" -c 'import sys; raise SystemExit(not ((3, 12) <= sys.version_info[:2] < (3, 15)))' \
            >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Splash needs Python 3.12, 3.13, or 3.14." >&2
    exit 1
fi

# The HF repo ships the release zip; extract it once into this directory:
#   hf download $HF_REPO
#   unzip <snapshot>/Qwen3.8-3.6-27B-blend-Splash-v0.3.zip -d ~/IdeaProjects/qwen3.8/models
PACKAGE="${SPLASH_PACKAGE:-$HOME/IdeaProjects/qwen3.8/models/Qwen3.8-3.6-27B-blend-Splash-v0.3}"

for path in "$BINARY" "$PACKAGE/manifest.json" "$PACKAGE/target" "$PACKAGE/draft" \
    "$PACKAGE/vision" "$PACKAGE/tokenizer"; do
    if [ ! -e "$path" ]; then
        echo "Missing Splash runtime/package path: $path" >&2
        exit 1
    fi
done

# The API model ID comes from the package manifest, as in the gateway.
MODEL_ID="$("$PYTHON" -c 'import json, sys
print(json.load(open(sys.argv[1]))["model"])' "$PACKAGE/manifest.json")"

echo "Starting Splash server with model: $MODEL_ID"
echo "Model location: $PACKAGE"
echo ""

export HF_HUB_OFFLINE=1

exec "$PYTHON" -u "$SCRIPT_DIR/server/server.py" \
    "$PACKAGE/target" \
    "$PACKAGE/draft" \
    --tokenizer "$PACKAGE/tokenizer" \
    --model "$MODEL_ID" \
    --binary "$BINARY" \
    "$@"
