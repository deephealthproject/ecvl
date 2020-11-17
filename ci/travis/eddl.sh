#!/bin/bash

# Eigen
if [ "$TRAVIS_OS" != "Darwin" ]; then
    if [ -d "${DEPS_INSTALL_DIR}/eigen" ]; then
        echo  "Eigen already built"
    else
        cd ${DEPS_BUILD_DIR}
        git clone https://gitlab.com/libeigen/eigen.git
        cd eigen
        if [ "$TRAVIS_OS" == "Windows" ]; then
            git checkout ba9d18b9388acdf27a3900a4f981fab587e59b0c # For VS2019
        else
            git checkout tags/3.3.7
        fi
        mkdir -p build && cd build
        cmake -G"${CMAKE_GENERATOR}" -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}/eigen \
            ..
        cmake --build . --config $BUILD_TYPE --parallel $PROC
        cmake --build . --config $BUILD_TYPE --target install
    fi

    export Eigen3_DIR=${DEPS_INSTALL_DIR}/eigen/share/eigen3/cmake

    # Protobuf
    if [ -d "${DEPS_INSTALL_DIR}/protobuf" ]; then
        echo  "Protobuf already built"
    else
        cd ${DEPS_BUILD_DIR}
        PB_VERSION=3.14.0
        wget -nv --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v$PB_VERSION/protobuf-cpp-$PB_VERSION.tar.gz
        tar -xzf protobuf-cpp-$PB_VERSION.tar.gz
        rm protobuf-cpp-$PB_VERSION.tar.gz
        mv protobuf-$PB_VERSION protobuf
        cd protobuf
        mkdir -p build_dir && cd build_dir
        cmake -G"${CMAKE_GENERATOR}" -Dprotobuf_BUILD_TESTS=OFF \
            -Dprotobuf_WITH_ZLIB=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=ON -Dprotobuf_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}/protobuf \
            ../cmake
        cmake --build . --config $BUILD_TYPE --parallel $PROC
        cmake --build . --config $BUILD_TYPE --target install
    fi

    export Protobuf_INCLUDE_DIR=${DEPS_INSTALL_DIR}/protobuf/include
    export Protobuf_INCLUDE_DIRS=${DEPS_INSTALL_DIR}/protobuf/include
    export Protobuf_LIBRARIES_DIRS=${DEPS_INSTALL_DIR}/protobuf/lib
    if [ "$TRAVIS_OS" == "Windows" ]; then
        export Protobuf_DIR=${DEPS_INSTALL_DIR}/protobuf/cmake
        export Protobuf_LIBRARIES=${Protobuf_LIBRARIES_DIRS}/libprotobuf.lib
        export Protobuf_LIBRARY_RELEASE=${Protobuf_LIBRARIES_DIRS}/libprotobuf.lib
        export Protobuf_LIBRARY_DEBUG=${Protobuf_LIBRARIES_DIRS}/libprotobuf.lib
        export Protobuf_PROTOC_EXECUTABLE=${DEPS_INSTALL_DIR}/protobuf/bin/protoc.exe
    else
        export Protobuf_DIR=${DEPS_INSTALL_DIR}/protobuf/lib/cmake/protobuf
        export Protobuf_LIBRARIES=${Protobuf_LIBRARIES_DIRS}/libprotobuf.a
        export Protobuf_LIBRARY_RELEASE=${Protobuf_LIBRARIES_DIRS}/libprotobuf.a
        export Protobuf_LIBRARY_DEBUG=${Protobuf_LIBRARIES_DIRS}/libprotobuf.a
        export Protobuf_PROTOC_EXECUTABLE=${DEPS_INSTALL_DIR}/protobuf/bin/protoc
    fi
else
    brew install eigen protobuf
fi


# EDDL
if [ -d "${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION}" ]; then
    echo "EDDL already installed"
else
    cd ${DEPS_BUILD_DIR}
    wget -nv --no-check-certificate https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
    mkdir -p eddl-$EDDL_VERSION && tar -xzf $EDDL_VERSION.tar.gz -C eddl-$EDDL_VERSION --strip-components 1
    rm $EDDL_VERSION.tar.gz
    cd eddl-${EDDL_VERSION}
    mkdir -p build && cd build

    if [ "$TRAVIS_OS" != "Darwin" ]; then
        cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION} \
            -DBUILD_TARGET=CPU -DBUILD_PROTOBUF=ON -DBUILD_TESTS=OFF \
            -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_SUPERBUILD=OFF \
            -DProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR} -DProtobuf_LIBRARY_RELEASE=${Protobuf_LIBRARY_RELEASE} \
            -DProtobuf_LIBRARY_DEBUG=${Protobuf_LIBRARY_DEBUG} -DProtobuf_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE} \
            ..
    else
        cmake -G"${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION} \
            -DBUILD_TARGET=CPU -DBUILD_PROTOBUF=ON -DBUILD_TESTS=OFF \
            -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_SUPERBUILD=OFF \
            ..
    fi
    cmake --build . --config $BUILD_TYPE --parallel $PROC
    cmake --build . --config $BUILD_TYPE --target install
fi

export eddl_DIR=${DEPS_INSTALL_DIR}/eddl-${EDDL_VERSION}/lib/cmake/eddl

# don't forget to switch back to the main build directory once you are done
cd ${TRAVIS_BUILD_DIR}