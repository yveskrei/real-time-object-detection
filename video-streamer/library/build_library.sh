#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$PROJECT_ROOT/dependencies"
FFMPEG_DIR="$DEPS_DIR/ffmpeg"

if [ ! -d "$FFMPEG_DIR" ]; then
  echo "❌ Error: FFmpeg not found. Run build_dependencies.sh first"
  exit 1
fi

cd "$PROJECT_ROOT"

echo "Building Rust library with static FFmpeg and XCB..."

export PKG_CONFIG_PATH="$FFMPEG_DIR/lib/pkgconfig:$DEPS_DIR/lib/pkgconfig:$DEPS_DIR/share/pkgconfig"
export PKG_CONFIG_STATIC=1
export FFMPEG_DIR="$FFMPEG_DIR"
export DEPS_DIR="$DEPS_DIR"
export PROJECT_ROOT="$PROJECT_ROOT"
export FFMPEG_INCLUDE_DIR="$FFMPEG_DIR/include"
export FFMPEG_LIB_DIR="$FFMPEG_DIR/lib"
export BINDGEN_EXTRA_CLANG_ARGS="-I$FFMPEG_DIR/include -I$DEPS_DIR/include"

export RUSTFLAGS="-C link-args=-Wl,--whole-archive \
-L$FFMPEG_DIR/lib -lavcodec -lavformat -lavutil -lavfilter -lswscale -lswresample -lpostproc \
-L$DEPS_DIR/lib -lx264 -lx265 -lvpx -lopus -lmp3lame \
-Wl,--no-whole-archive \
-L$DEPS_DIR/lib -lsrt -lssl -lcrypto -lxcb -lxcb-shm -lxcb-shape -lxcb-xfixes -lXau -lXdmcp -llzma -lbz2 \
-lstdc++ -lm -lz -lpthread -ldl"

cargo clean
cargo build --release

echo "✅ Build complete!"