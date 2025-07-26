FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# Set non-interactive mode for apt to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Run the scripts
RUN python3 -m pip install --upgrade pip && /opt/nvidia/deepstream/deepstream-7.1/install.sh
RUN /opt/nvidia/deepstream/deepstream-7.1/user_additional_install.sh
RUN /opt/nvidia/deepstream/deepstream-7.1/user_deepstream_python_apps_install.sh

# Update apt and install required packages
RUN apt-get update && apt-get install -y \
    libflac8 \
    libfaad2 \
    libopenh264-6 \
    libvo-aacenc0 \
    libmjpegtools-dev \
    mjpegtools \
    ubuntu-restricted-extras \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*