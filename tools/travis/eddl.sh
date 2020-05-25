#!/bin/bash

if [[ "$TRAVIS_OS" == "Linux" ]]; then
    sudo apt-get install -y libeigen3-dev zlib1g-dev
elif [[ "$TRAVIS_OS" == "Darwin" ]]; then
    brew install eigen zlib
fi

# Protobuf
if [ -d "${DEPS_INSTALL_DIR}/protobuf" ]; then
    echo -e "Protobuf already built"
else
    mkdir -p ${DEPS_INSTALL_DIR} && cd ${DEPS_INSTALL_DIR}
    wget --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v3.12.1/protobuf-cpp-3.12.1.tar.gz
    tar -xzf protobuf-cpp-3.12.1.tar.gz
    rm protobuf-cpp-3.12.1.tar.gz
    mv protobuf-3.12.1 protobuf
    cd protobuf
    mkdir -p build && cd build
    cmake -G "Unix Makefiles" -Dprotobuf_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr ../cmake
    cmake --build . --config $BUILD_TYPE --parallel $PROC
fi

# Install Protobuf
cd ${DEPS_INSTALL_DIR}/protobuf
# Use cmake absolute path for sudo
sudo ${CMAKE_BIN} --build build --config $BUILD_TYPE --target install
sudo ldconfig

if [ -d "${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION}" ]; then
    echo -e "EDDL already installed"
else
    mkdir -p ${DEPS_BUILD_DIR} && cd ${DEPS_BUILD_DIR}
    wget --no-check-certificate https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
    tar -xzf $EDDL_VERSION.tar.gz
    cd eddl-${EDDL_VERSION}
    mkdir build && cd build

    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION} \
        -DBUILD_TARGET=CPU -DBUILD_PROTOBUF=ON -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_SUPERBUILD=OFF \
        ..
    cmake --build . --config $BUILD_TYPE --parallel $PROC
    cmake --build . --config $BUILD_TYPE --target install
fi

export eddl_DIR=${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION}

# don't forget to switch back to the main build directory once you are done
cd ${TRAVIS_BUILD_DIR}