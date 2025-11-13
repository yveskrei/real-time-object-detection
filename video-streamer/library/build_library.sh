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

# Critical: Use -Wl,-Bstatic for our static libs, then -Wl,-Bdynamic for system libs
# This ensures system libraries like libc are dynamically linked
export RUSTFLAGS="-C link-arg=-L$FFMPEG_DIR/lib \
-C link-arg=-L$DEPS_DIR/lib \
-C link-arg=-L$DEPS_DIR/lib64 \
-C link-arg=-Wl,-Bstatic \
-C link-arg=-lpostproc \
-C link-arg=-lswresample \
-C link-arg=-lswscale \
-C link-arg=-lavfilter \
-C link-arg=-lavformat \
-C link-arg=-lavcodec \
-C link-arg=-lavutil \
-C link-arg=-lsrt \
-C link-arg=-lssl \
-C link-arg=-lcrypto \
-C link-arg=-lx264 \
-C link-arg=-lx265 \
-C link-arg=-lvpx \
-C link-arg=-lopus \
-C link-arg=-lmp3lame \
-C link-arg=-lxcb \
-C link-arg=-lxcb-shm \
-C link-arg=-lxcb-shape \
-C link-arg=-lxcb-xfixes \
-C link-arg=-lXau \
-C link-arg=-lXdmcp \
-C link-arg=-llzma \
-C link-arg=-lbz2 \
-C link-arg=-Wl,-Bdynamic \
-C link-arg=-lstdc++ \
-C link-arg=-lm \
-C link-arg=-lz \
-C link-arg=-lpthread \
-C link-arg=-ldl \
-C link-arg=-lc"

cargo clean
cargo build --release

echo "✅ Build complete!"