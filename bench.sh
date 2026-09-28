#!/usr/bin/env bash
set -euo pipefail

# Benchmark the running Splash server by replaying a captured real Junie
# session (dev/benchmarks/junie_replay). Prints per-request serving stats —
# KV cached, prefill speed over new tokens, generation speed, DFlash
# acceptance — and their means, in the same form as mlx-vlm's bench.sh.
#
# Usage:
#   ./bench.sh                              # server on 127.0.0.1:8000
#   PORT=8085 ./bench.sh                    # another port
#   ./bench.sh --reasoning-effort low       # extra flags go to replay.py
#   ./bench.sh --model NAME                 # override the served model name
#
# A server with an API key needs it in SPLASH_API_KEY.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: python3 is missing." >&2
    exit 1
  fi
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://$HOST:$PORT"

HEALTH_CURL=(curl -sf -m 5)
if [ -n "${SPLASH_API_KEY:-}" ]; then
  HEALTH_CURL+=(-H "Authorization: Bearer $SPLASH_API_KEY")
fi
if ! "${HEALTH_CURL[@]}" "$URL/ready" > /dev/null 2>&1; then
  echo "Splash server is not ready on $URL."
  echo "Start it first:  ./start_splash.sh   (or: splash serve --model ...)"
  exit 1
fi

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/dev/benchmarks/junie_replay/replay.py" \
  --url "$URL" "$@"
