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

# Export paths to our self-built binaries and config files
export PATH="$DEPS_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$DEPS_DIR/lib/pkgconfig:$DEPS_DIR/lib64/pkgconfig:$DEPS_DIR/share/pkgconfig:$PKG_CONFIG_PATH"
export ACLOCAL_PATH="$DEPS_DIR/share/aclocal:$ACLOCAL_PATH"

cd "$DEPS_DIR"

# Download and build libbz2 (bzip2)
echo "Downloading libbz2..."
git clone https://sourceware.org/git/bzip2.git
echo "Building libbz2..."
cd bzip2
make -j"$NPROC" CFLAGS="-fPIC -O2 -g -D_FILE_OFFSET_BITS=64"
make install PREFIX="$DEPS_DIR"
cd ..

# Download and build gettext (for autopoint, required by liblzma)
echo "Downloading gettext..."
curl -OL https://ftp.gnu.org/pub/gnu/gettext/gettext-0.22.5.tar.gz
tar -xzf gettext-0.22.5.tar.gz
echo "Building gettext (for autopoint)..."
cd gettext-0.22.5
./configure --prefix="$DEPS_DIR" --disable-shared --enable-static --with-pic --without-git
make -j"$NPROC"
make install
cd ..

# Download and build liblzma (XZ Utils)
echo "Downloading liblzma..."
git clone https://git.tukaani.org/xz.git
echo "Building liblzma..."
cd xz
./autogen.sh || true
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic --disable-doc
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Download and build zlib
echo "Downloading zlib..."
curl -OL https://www.zlib.net/zlib-1.3.1.tar.gz
tar -xzf zlib-1.3.1.tar.gz
echo "Building zlib..."
cd zlib-1.3.1
CFLAGS="-fPIC" ./configure --prefix="$DEPS_DIR" --static
make -j"$NPROC"
make install
cd ..

# Download and build xorg-macros (required by libXau, etc.)
echo "Downloading xorg-macros..."
git clone https://gitlab.freedesktop.org/xorg/util/macros.git
echo "Building xorg-macros..."
cd macros
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Download and build xorgproto
echo "Downloading xorgproto..."
git clone https://gitlab.freedesktop.org/xorg/proto/xorgproto.git
echo "Building xorgproto..."
cd xorgproto
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Download and build xcb-proto (required by libxcb)
echo "Downloading xcb-proto..."
git clone https://gitlab.freedesktop.org/xorg/proto/xcbproto.git
echo "Building xcb-proto..."
cd xcbproto
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Download and build libXau (required by libxcb)
echo "Downloading libXau..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxau.git
echo "Building libXau..."
cd libxau
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Download and build libXdmcp (required by libxcb)
echo "Downloading libXdmcp..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxdmcp.git
echo "Building libXdmcp..."
cd libxdmcp
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Download and build libxcb
echo "Downloading libxcb..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxcb.git
echo "Building libxcb..."
cd libxcb
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Download and build nasm (required by x264, x265)
echo "Downloading nasm..."
curl -OL https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/nasm-2.16.01.tar.gz
tar -xzf nasm-2.16.01.tar.gz
echo "Building nasm..."
cd nasm-2.16.01
./configure --prefix="$DEPS_DIR"
make -j"$NPROC"
make install
cd ..

# Download and build libmp3lame
echo "Downloading libmp3lame..."
git clone https://github.com/lameproject/lame.git
echo "Building libmp3lame..."
cd lame
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Download and build libopus
echo "Downloading libopus..."
git clone https://github.com/xiph/opus.git
cd opus
# Download the model file that opus tries to fetch during build
mkdir -p models
curl -L -o models/opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz \
  https://media.xiph.org/opus/models/opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz
echo "Building libopus..."
# Use autoreconf instead of autogen.sh to avoid downloading models
autoreconf -fiv
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic --disable-intrinsics
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Download and build libvpx
echo "Downloading libvpx..."
git clone https://chromium.googlesource.com/webm/libvpx.git
echo "Building libvpx..."
cd libvpx
./configure --prefix="$DEPS_DIR" --disable-shared --enable-static --enable-pic
make -j"$NPROC"
make install
cd ..

# Download and build x264
echo "Downloading x264..."
git clone https://code.videolan.org/videolan/x264.git
echo "Building x264..."
cd x264
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --enable-pic
make -j"$NPROC"
make install
cd ..

# Download and build x265
echo "Downloading x265..."
git clone https://bitbucket.org/multicoreware/x265_git.git
echo "Building x265..."
mkdir -p x265_git/build
cd x265_git/build

cmake -G "Unix Makefiles" \
  -DENABLE_SHARED=OFF \
  -DENABLE_CLI=OFF \
  -DCMAKE_INSTALL_PREFIX="$DEPS_DIR" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DNASM_EXECUTABLE="$DEPS_DIR/bin/nasm" \
  -DENABLE_PKGCONFIG=ON \
  ../source
  
make -j"$NPROC"
make install

# We add -lstdc++ to Libs.private so pkg-config can find it.
echo "Manually creating x265.pc..."
mkdir -p "$DEPS_DIR/lib/pkgconfig"
tee "$DEPS_DIR/lib/pkgconfig/x265.pc" > /dev/null <<EOF
prefix=$DEPS_DIR
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: x265
Description: H.265/HEVC video encoder
Version: 3.5
Libs: -L\${libdir} -lx265
Libs.private: -lstdc++ -lpthread -ldl -lm
Cflags: -I\${includedir}
EOF

cd ../../

# Download and build OpenSSL (dependency for libsrt)
echo "Downloading OpenSSL..."
git clone --depth 1 --branch openssl-3.1.4 https://github.com/openssl/openssl.git
echo "Building OpenSSL..."
cd openssl
./config --prefix="$DEPS_DIR" --openssldir="$DEPS_DIR/ssl" no-shared no-tests -fPIC
make -j"$NPROC"
make install_sw
cd ..

# Download and build libsrt
echo "Downloading libsrt..."
git clone --depth 1 --branch v1.5.3 https://github.com/Haivision/srt.git
echo "Building libsrt..."
cd srt
mkdir -p build && cd build
# Added -DENABLE_GCRYPT=OFF to force use of OpenSSL
cmake .. -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="$DEPS_DIR" \
  -DENABLE_SHARED=OFF \
  -DENABLE_STATIC=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DENABLE_APPS=OFF \
  -DOPENSSL_USE_STATIC_LIBS=ON \
  -DOPENSSL_ROOT_DIR="$DEPS_DIR" \
  -DCMAKE_PREFIX_PATH="$DEPS_DIR" \
  -DENABLE_GCRYPT=OFF
make -j"$NPROC"
make install
cd ../..

# Download and build FFmpeg 6.1
echo "Downloading FFmpeg 6.1..."
git clone --depth 1 --branch n6.1 https://github.com/FFmpeg/FFmpeg.git "$FFMPEG_SRC"
echo "Building FFmpeg 6.1..."
cd "$FFMPEG_SRC"

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
  --enable-openssl \
  --enable-libsrt \
  --extra-cflags="-fPIC -I$DEPS_DIR/include" \
  --extra-cxxflags="-fPIC -I$DEPS_DIR/include" \
  --extra-ldflags="-L$DEPS_DIR/lib -L$DEPS_DIR/lib64"

make -j"$NPROC"
make install

# Compress only the required folders
echo "Compressing dependencies..."
cd "$PROJECT_ROOT"
tar -czf dependencies.tar.gz \
  dependencies/lib \
  dependencies/lib64 \
  dependencies/ffmpeg

echo "✅ All dependencies built and compressed to $PROJECT_ROOT/dependencies.tar.gz"