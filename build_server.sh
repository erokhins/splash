#!/usr/bin/env bash
set -euo pipefail

# Freeze server/server.py and the native engine into a self-contained tar.gz.
#
#   Output:  dist/splash-server-<version>-macos-arm64.tar.gz
#   Run:     splash-server/splash-server <target> <draft> --tokenizer <dir> --model <id> [server.py args]
#
# Takes the same arguments as server/server.py (see start_splash.sh); --binary
# defaults to the bundled engine. Models are not bundled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NAME="splash-server"
VERSION="$(git describe --tags --always --dirty)"
ARCHIVE="$SCRIPT_DIR/dist/$NAME-$VERSION-macos-arm64.tar.gz"

# The engine and its shader library, built as `make all` builds them for the
# source checkout. The engine opens splash.metallib beside its own executable.
make platform-check
make -j4 all

# Deleted after the archive is built; a failed build leaves it for inspection.
BUILD_DIR="$(mktemp -d -t "build-$NAME")"
PYTHON="$BUILD_DIR/.venv/bin/python"

# uv installs a managed CPython; Splash supports 3.12 to 3.14.
UV_PYTHON_INSTALL_DIR="$BUILD_DIR/.python" \
  uv venv --managed-python --python 3.13 "$BUILD_DIR/.venv"
uv pip install --python "$PYTHON" -r install/requirements.txt pyinstaller

# PyInstaller follows every import statement, including those inside function
# bodies. These are the files it cannot see that way.
HIDDEN_DEPS=(
  # The engine and its shader library, kept together (see above).
  --add-binary "$SCRIPT_DIR/build/splash:engine"
  --add-data "$SCRIPT_DIR/build/splash.metallib:engine"
  # server.py reads the chat page beside itself.
  --add-data "$SCRIPT_DIR/server/chat.html:server"
  # server.py imports its siblings relatively only when run as a package,
  # which the entry point does; collect them all by package name.
  --collect-submodules server
  # llguidance/__init__.py does `__version__ = version("llguidance")` at
  # import time, so it needs its .dist-info, not only its code.
  --copy-metadata llguidance
)

# --noupx: UPX breaks macOS binaries; PyInstaller uses it if present (upx#601).
# --paths: the entry point imports `server` from the checkout root.
# pushd: PyInstaller writes dist/, build/ and the spec into the cwd.
pushd "$BUILD_DIR"
"$PYTHON" -m PyInstaller \
  --onedir --name "$NAME" \
  --noupx \
  --console --target-arch arm64 \
  --paths "$SCRIPT_DIR" \
  "${HIDDEN_DEPS[@]}" \
  "$SCRIPT_DIR/dev/tools/frozen_server.py"
popd

BUNDLE="$BUILD_DIR/dist/$NAME"
"$BUNDLE/$NAME" --help >/dev/null
# The engine starts from its bundled location and finds its shader library.
"$BUNDLE/_internal/engine/splash" device-check

mkdir -p "$(dirname "$ARCHIVE")"
tar -czf "$ARCHIVE" -C "$BUILD_DIR/dist" "$NAME"

# pushd: keeps the path inside the .sha256 file relative.
pushd "$(dirname "$ARCHIVE")"
shasum -a 256 "$(basename "$ARCHIVE")" | tee "$(basename "$ARCHIVE").sha256"
popd

rm -rf "$BUILD_DIR"
echo "Built $ARCHIVE"
