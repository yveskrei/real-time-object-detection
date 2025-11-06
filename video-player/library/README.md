# Video Client Library
This library provides a user friendly integration for consuming real time video streams. It allows to send back AI analytics results and receive video streams with low latency.
The library is built using **Rust** language and provides C bindings for easy integration with third party applications.

## Building the Library
To build the library, ensure you have the necessary dependencies installed. You can then run the following command in the terminal:
```bash
# Install dependencies on Fedora-based systems
dnf install -y \
  gcc \
  gcc-c++ \
  make \
  cmake \
  git \
  autoconf \
  automake \
  libtool \
  pkgconfig \
  perl \
  python3 \
  diffutils \
  gettext \
  wget

# Build dependencies required to build the library
./build_dependencies.sh

# Build the library
./build_library.sh
```