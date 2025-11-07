#!/usr/bin/env bash
set -e

NPROC=$(nproc)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$PROJECT_ROOT/dependencies"
FFMPEG_DIR="$DEPS_DIR/ffmpeg"
FFMPEG_SRC="$FFMPEG_DIR/src"

# Export paths to our self-built binaries and config files
export PATH="$DEPS_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$DEPS_DIR/lib/pkgconfig:$DEPS_DIR/lib64/pkgconfig:$DEPS_DIR/share/pkgconfig:$PKG_CONFIG_PATH"
export ACLOCAL_PATH="$DEPS_DIR/share/aclocal:$ACLOCAL_PATH"

if [ -d "$DEPS_DIR" ]; then
  echo "Removing existing dependencies..."
  rm -rf "$DEPS_DIR"
fi
mkdir -p "$DEPS_DIR"

cd "$DEPS_DIR"

# Build libbz2 (bzip2)
echo "Building libbz2..."
git clone https://sourceware.org/git/bzip2.git
cd bzip2
make -j"$NPROC" CFLAGS="-fPIC -O2 -g -D_FILE_OFFSET_BITS=64"
make install PREFIX="$DEPS_DIR"
cd ..

# Build gettext (for autopoint, required by liblzma)
echo "Building gettext (for autopoint)..."
curl -OL https://ftp.gnu.org/pub/gnu/gettext/gettext-0.22.5.tar.gz
tar -xzf gettext-0.22.5.tar.gz
cd gettext-0.22.5
./configure --prefix="$DEPS_DIR" --disable-shared --enable-static --with-pic --without-git
make -j"$NPROC"
make install
cd ..

# Build liblzma (XZ Utils)
echo "Building liblzma..."
git clone https://git.tukaani.org/xz.git
cd xz
./autogen.sh || true
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic --disable-doc
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build xorg-macros (required by libXau, etc.)
echo "Building xorg-macros..."
git clone https://gitlab.freedesktop.org/xorg/util/macros.git
cd macros
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Build xorgproto
echo "Building xorgproto..."
git clone https://gitlab.freedesktop.org/xorg/proto/xorgproto.git
cd xorgproto
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Build xcb-proto (required by libxcb)
echo "Building xcb-proto..."
git clone https://gitlab.freedesktop.org/xorg/proto/xcbproto.git
cd xcbproto
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Build libXau (required by libxcb)
echo "Building libXau..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxau.git
cd libxau
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libXdmcp (required by libxcb)
echo "Building libXdmcp..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxdmcp.git
cd libxdmcp
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libxcb
echo "Building libxcb..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxcb.git
cd libxcb
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build nasm (required by x264, x265)
echo "Building nasm..."
curl -OL https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/nasm-2.16.01.tar.gz
tar -xzf nasm-2.16.01.tar.gz
cd nasm-2.16.01
./configure --prefix="$DEPS_DIR"
make -j"$NPROC"
make install
cd ..

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
  -DENABLE_PKGCONFIG=ON \
  ../source
make -j"$NPROC"
make install
cd ../../

# Build FFmpeg 6.1
echo "Building FFmpeg 6.1..."
git clone --depth 1 --branch n6.1 https://github.com/FFmpeg/FFmpeg.git "$FFMPEG_SRC"
cd "$FFMPEG_SRC"

# Note: PKG_CONFIG_PATH is set at the top of the script
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
  --extra-cflags="-fPIC -I$DEPS_DIR/include" \
  --extra-cxxflags="-fPIC -I$DEPS_DIR/include" \
  --extra-ldflags="-L$DEPS_DIR/lib -L$DEPS_DIR/lib64"

make -j"$NPROC"
make install

echo "✅ Dependencies build complete."