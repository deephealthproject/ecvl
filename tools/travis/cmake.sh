#!/bin/bash

if [ -d "${DEPS_INSTALL_DIR}/cmake" ]; then
    echo -e "CMake already installed"
else
    # Retrieve CMake 3.13.5
    mkdir -p ${DEPS_INSTALL_DIR} && cd ${DEPS_INSTALL_DIR}
    wget -t 3 --no-check-certificate https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.tar.gz
    # do a quick sha256 check to ensure that the archive we downloaded did not get compromised
    echo "e2fd0080a6f0fc1ec84647acdcd8e0b4019770f48d83509e6a5b0b6ea27e5864 *cmake-3.13.5-Linux-x86_64.tar.gz" > cmake_sha256.txt
    sha256sum -c cmake_sha256.txt
    tar -xf cmake-3.13.5-Linux-x86_64.tar.gz
    mv cmake-3.13.5-Linux-x86_64 cmake
fi 

# Add CMake to path
export PATH=${DEPS_INSTALL_DIR}/cmake/bin:$PATH

# don't forget to switch back to the main build directory once you are done
cd ${TRAVIS_BUILD_DIR}