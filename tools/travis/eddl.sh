#!/bin/bash

if [[ "$TRAVIS_OS" == "Linux" ]]; then
    sudo apt-get install libeigen3-dev zlib1g-dev
elif [[ "$TRAVIS_OS" == "Darwin" ]]; then
    brew install eigen zlib
fi
# Protobuf
wget --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v3.12.1/protobuf-cpp-3.12.1.tar.gz
tar -xzf protobuf-cpp-3.12.1.tar.gz
cd protobuf-3.12.1 && ./configure --prefix=/usr && make && sudo make install
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