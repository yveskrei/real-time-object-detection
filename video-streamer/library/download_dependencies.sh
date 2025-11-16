#!/usr/bin/env bash
set -e

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

# Download libbz2 (bzip2)
echo "Downloading libbz2..."
git clone https://sourceware.org/git/bzip2.git

# Download gettext (for autopoint, required by liblzma)
echo "Downloading gettext..."
curl -OL https://ftp.gnu.org/pub/gnu/gettext/gettext-0.22.5.tar.gz
tar -xzf gettext-0.22.5.tar.gz

# Download liblzma (XZ Utils)
echo "Downloading liblzma..."
git clone https://git.tukaani.org/xz.git

# Download zlib
echo "Downloading zlib..."
curl -OL https://www.zlib.net/zlib-1.3.1.tar.gz
tar -xzf zlib-1.3.1.tar.gz

# Download xorg-macros (required by libXau, etc.)
echo "Downloading xorg-macros..."
git clone https://gitlab.freedesktop.org/xorg/util/macros.git

# Download xorgproto
echo "Downloading xorgproto..."
git clone https://gitlab.freedesktop.org/xorg/proto/xorgproto.git

# Download xcb-proto (required by libxcb)
echo "Downloading xcb-proto..."
git clone https://gitlab.freedesktop.org/xorg/proto/xcbproto.git

# Download libXau (required by libxcb)
echo "Downloading libXau..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxau.git

# Download libXdmcp (required by libxcb)
echo "Downloading libXdmcp..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxdmcp.git

# Download libxcb
echo "Downloading libxcb..."
git clone https://gitlab.freedesktop.org/xorg/lib/libxcb.git

# Download nasm (required by x264, x265)
echo "Downloading nasm..."
curl -OL https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/nasm-2.16.01.tar.gz
tar -xzf nasm-2.16.01.tar.gz

# Download libmp3lame
echo "Downloading libmp3lame..."
git clone https://github.com/lameproject/lame.git

# Download libopus
echo "Downloading libopus..."
git clone https://github.com/xiph/opus.git
cd opus
# Download the model file that opus tries to fetch during build
mkdir -p models
curl -L -o models/opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz \
  https://media.xiph.org/opus/models/opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz
cd ..

# Download libvpx
echo "Downloading libvpx..."
git clone https://chromium.googlesource.com/webm/libvpx.git

# Download x264
echo "Downloading x264..."
git clone https://code.videolan.org/videolan/x264.git

# Download x265
echo "Downloading x265..."
git clone https://bitbucket.org/multicoreware/x265_git.git

# Download OpenSSL (dependency for libsrt)
echo "Downloading OpenSSL..."
git clone --depth 1 --branch openssl-3.1.4 https://github.com/openssl/openssl.git

# Download libsrt
echo "Downloading libsrt..."
git clone --depth 1 --branch v1.5.3 https://github.com/Haivision/srt.git

# Download FFmpeg 6.1
echo "Downloading FFmpeg 6.1..."
git clone --depth 1 --branch n6.1 https://github.com/FFmpeg/FFmpeg.git "$FFMPEG_SRC"

echo "Compressing dependencies..."
cd "$PROJECT_ROOT"
tar -czf dependencies.tar.gz dependencies/

echo "✅ All dependencies downloaded and compressed to $PROJECT_ROOT/dependencies.tar.gz"
echo "You can now transfer dependencies.tar.gz and run build_dependencies.sh for offline building"