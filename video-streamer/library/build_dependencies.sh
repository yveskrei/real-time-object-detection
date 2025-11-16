#!/usr/bin/env bash
set -e

NPROC=$(nproc)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_FILE="$PROJECT_ROOT/dependencies.tar.gz"
DEPS_DIR="$PROJECT_ROOT/dependencies"
FFMPEG_DIR="$DEPS_DIR/ffmpeg"
FFMPEG_SRC="$FFMPEG_DIR/src"

# Check if archive exists
if [ ! -f "$ARCHIVE_FILE" ]; then
  echo "ERROR: Dependencies archive not found at $ARCHIVE_FILE"
  echo "Please run download_dependencies.sh first"
  exit 1
fi

# Extract dependencies archive
echo "Extracting dependencies archive..."
if [ -d "$DEPS_DIR" ]; then
  rm -rf "$DEPS_DIR"
fi
tar -xzf "$ARCHIVE_FILE" -C "$PROJECT_ROOT"

# Export paths to our self-built binaries and config files
export PATH="$DEPS_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$DEPS_DIR/lib/pkgconfig:$DEPS_DIR/lib64/pkgconfig:$DEPS_DIR/share/pkgconfig:$PKG_CONFIG_PATH"
export ACLOCAL_PATH="$DEPS_DIR/share/aclocal:$ACLOCAL_PATH"

cd "$DEPS_DIR"

# Build libbz2 (bzip2)
echo "Building libbz2..."
cd bzip2
make -j"$NPROC" CFLAGS="-fPIC -O2 -g -D_FILE_OFFSET_BITS=64"
make install PREFIX="$DEPS_DIR"
cd ..

# Build gettext (for autopoint, required by liblzma)
echo "Building gettext (for autopoint)..."
cd gettext-0.22.5
./configure --prefix="$DEPS_DIR" --disable-shared --enable-static --with-pic --without-git
make -j"$NPROC"
make install
cd ..

# Build liblzma (XZ Utils)
echo "Building liblzma..."
cd xz
./autogen.sh || true
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic --disable-doc
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build zlib
echo "Building zlib..."
cd zlib-1.3.1
# Pass -fPIC via CFLAGS environment variable
CFLAGS="-fPIC" ./configure --prefix="$DEPS_DIR" --static
make -j"$NPROC"
make install
cd ..

# Build xorg-macros (required by libXau, etc.)
echo "Building xorg-macros..."
cd macros
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Build xorgproto
echo "Building xorgproto..."
cd xorgproto
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Build xcb-proto (required by libxcb)
echo "Building xcb-proto..."
cd xcbproto
./autogen.sh
./configure --prefix="$DEPS_DIR"
make install
cd ..

# Build libXau (required by libxcb)
echo "Building libXau..."
cd libxau
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libXdmcp (required by libxcb)
echo "Building libXdmcp..."
cd libxdmcp
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libxcb
echo "Building libxcb..."
cd libxcb
./autogen.sh
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build nasm (required by x264, x265)
echo "Building nasm..."
cd nasm-2.16.01
./configure --prefix="$DEPS_DIR"
make -j"$NPROC"
make install
cd ..

# Build libmp3lame
echo "Building libmp3lame..."
cd lame
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libopus
echo "Building libopus..."
cd opus
# Use autoreconf instead of autogen.sh to avoid downloading models
autoreconf -fiv
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --with-pic --disable-intrinsics
make -j"$NPROC" CFLAGS="-fPIC"
make install
cd ..

# Build libvpx
echo "Building libvpx..."
cd libvpx
./configure --prefix="$DEPS_DIR" --disable-shared --enable-static --enable-pic
make -j"$NPROC"
make install
cd ..

# Build x264
echo "Building x264..."
cd x264
./configure --prefix="$DEPS_DIR" --enable-static --disable-shared --enable-pic
make -j"$NPROC"
make install
cd ..

# Build x265
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

# Build OpenSSL (dependency for libsrt)
echo "Building OpenSSL..."
cd openssl
./config --prefix="$DEPS_DIR" --openssldir="$DEPS_DIR/ssl" no-shared no-tests -fPIC
make -j"$NPROC"
make install_sw
cd ..

# Build libsrt
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

# Build FFmpeg 6.1
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

echo "✅ Dependencies build complete (offline mode)."