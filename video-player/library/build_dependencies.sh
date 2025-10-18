#!/usr/bin/env bash
set -e

NPROC=$(nproc)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$PROJECT_ROOT/dependencies"
FFMPEG_DIR="$DEPS_DIR/ffmpeg"
FFMPEG_SRC="$FFMPEG_DIR/src"

if [ -d "$DEPS_DIR" ]; then
  echo "Removing existing dependencies..."
  rm -rf "$DEPS_DIR"
fi
mkdir -p "$DEPS_DIR"

cd "$DEPS_DIR"

# Build libmp3lame
echo "Building libmp3lame..."
git clone https://github.com/lameproject/lame.git
cd lame
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libopus
echo "Building libopus..."
git clone https://github.com/xiph/opus.git
cd opus
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic --disable-intrinsics
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libvpx
echo "Building libvpx..."
git clone https://chromium.googlesource.com/webm/libvpx.git
cd libvpx
./configure --prefix="$DEPS_DIR" --disable-shared --enable-static --enable-pic
make -j"$NPROC"
make install
cd ..

# Build x264
echo "Building x264..."
git clone https://code.videolan.org/videolan/x264.git
cd x264
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --enable-pic
make -j"$NPROC"
make install
cd ..

# Build x265
echo "Building x265..."
git clone https://bitbucket.org/multicoreware/x265_git.git
cd x265_git/build
cmake -G "Unix Makefiles" \
  -DENABLE_SHARED=OFF \
  -DENABLE_CLI=OFF \
  -DCMAKE_INSTALL_PREFIX="$DEPS_DIR" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  ../source
make -j"$NPROC"
make install
cd ../../

# Build FFmpeg 6.1
echo "Building FFmpeg 6.1..."
git clone --depth 1 --branch n6.1 https://github.com/FFmpeg/FFmpeg.git "$FFMPEG_SRC"
cd "$FFMPEG_SRC"

export PKG_CONFIG_PATH="$DEPS_DIR/lib/pkgconfig"

./configure \
  --prefix="$FFMPEG_DIR" \
  --disable-shared \
  --enable-static \
  --enable-pic \
  --enable-gpl \
  --enable-version3 \
  --disable-doc \
  --disable-debug \
  --pkg-config-flags="--static" \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libvpx \
  --enable-libopus \
  --enable-libmp3lame \
  --extra-cflags="-fPIC" \
  --extra-cxxflags="-fPIC"

make -j"$NPROC"
make install

echo "✅ Dependencies build complete."