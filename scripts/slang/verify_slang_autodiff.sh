#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SLANGC_BIN="${SLANGC:-slangc}"
SLANG_PROFILE="${SLANG_PROFILE:-metal_2_4}"
SLANG_SOURCE="${SLANG_SOURCE:-$ROOT_DIR/slang/probes/autodiff_probe.slang}"
SLANG_OUT_DIR="${SLANG_OUT_DIR:-$ROOT_DIR/build/slang_verify}"

if [[ "$SLANG_SOURCE" != /* ]]; then
    SLANG_SOURCE="$ROOT_DIR/$SLANG_SOURCE"
fi
if [[ "$SLANG_OUT_DIR" != /* ]]; then
    SLANG_OUT_DIR="$ROOT_DIR/$SLANG_OUT_DIR"
fi

if [[ ! -f "$SLANG_SOURCE" ]]; then
    echo "error: Slang source not found: $SLANG_SOURCE" >&2
    exit 2
fi

if ! command -v "$SLANGC_BIN" >/dev/null 2>&1; then
    echo "error: '$SLANGC_BIN' was not found." >&2
    echo "Install Slang from https://github.com/shader-slang/slang/releases" >&2
    echo "or pass an explicit path: make slang-check SLANGC=/path/to/slangc" >&2
    exit 127
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "error: xcrun was not found. Install Xcode command line tools." >&2
    exit 127
fi

mkdir -p "$SLANG_OUT_DIR"

entries=(
    run_backward_auto
    run_backward_custom
)

for entry in "${entries[@]}"; do
    metal_src="$SLANG_OUT_DIR/$entry.metal"
    air_bin="$SLANG_OUT_DIR/$entry.air"
    metallib_bin="$SLANG_OUT_DIR/$entry.metallib"

    echo "[slang-check] compiling $entry -> $metal_src"
    "$SLANGC_BIN" "$SLANG_SOURCE" \
        -target metal \
        -profile "$SLANG_PROFILE" \
        -entry "$entry" \
        -stage compute \
        -o "$metal_src"

    echo "[slang-check] validating Metal compile: $metal_src"
    xcrun metal -std=metal3.1 -c "$metal_src" -o "$air_bin"
    xcrun metallib "$air_bin" -o "$metallib_bin"
done

echo "[slang-check] success"
echo "[slang-check] outputs:"
ls -1 "$SLANG_OUT_DIR"/*.metal "$SLANG_OUT_DIR"/*.metallib
