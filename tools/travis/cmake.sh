#!/bin/bash

if [ -d "${DEPS_INSTALL_DIR}/cmake" ]; then
    echo -e "CMake already installed"
else
    mkdir -p ${DEPS_INSTALL_DIR} && cd ${DEPS_INSTALL_DIR}
    # Retrieve CMake 3.13.5
    wget --no-check-certificate https://cmake.org/files/v3.13/cmake-3.13.5-$TRAVIS_OS-x86_64.tar.gz
    tar -xzf cmake-3.13.5-$TRAVIS_OS-x86_64.tar.gz
    mv cmake-3.13.5-$TRAVIS_OS-x86_64 cmake
fi

# Add CMake to path
if [[ "$TRAVIS_OS" == "Linux" ]]; then
    export PATH="${DEPS_INSTALL_DIR}/cmake/bin":${PATH}
    export CMAKE_BIN="${DEPS_INSTALL_DIR}/cmake/bin/cmake"
elif [[ "$TRAVIS_OS" == "Darwin" ]]; then
    export PATH="${DEPS_INSTALL_DIR}/cmake/CMake.app/Contents/bin":${PATH}
    export CMAKE_BIN="${DEPS_INSTALL_DIR}/cmake/CMake.app/Contents/bin/cmake"
fi

# don't forget to switch back to the main build directory once you are done
cd ${TRAVIS_BUILD_DIR}