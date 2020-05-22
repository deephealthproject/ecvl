#!/bin/bash

if [ -d "${DEPS_INSTALL_DIR}/opencv" ]; then
    echo -e "OpenCV already installed"
else
    mkdir -p ${DEPS_BUILD_DIR} && cd ${DEPS_BUILD_DIR}
    wget -t 3 --no-check-certificate https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz
    tar xf $OPENCV_VERSION.tar.gz
    mv opencv-$OPENCV_VERSION opencv
    cd opencv
    mkdir build && cd build

    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}/opencv \
        -DBUILD_LIST=core,imgproc,imgcodecs,photo,calib3d -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_java_bindings_generator=OFF -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_JPEG=ON \
        -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=ON -DBUILD_WEBP=OFF \
        -DBUILD_ZLIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DWITH_1394=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JPEG=ON -DWITH_LAPACK=OFF \
        -DWITH_MATLAB=OFF -DWITH_OPENCL=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF \
        -DWITH_PNG=ON -DWITH_PROTOBUF=OFF -DWITH_QT=OFF -DWITH_QUIRC=OFF \
        -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_V4L=OFF -DWITH_VTK=OFF -DWITH_WEBP=OFF \
        ..
    cmake --build . --config $BUILD_TYPE --parallel $PROC
    cmake --build . --config $BUILD_TYPE --target install
fi

export OpenCV_DIR=${DEPS_INSTALL_DIR}/opencv

# don't forget to switch back to the main build directory once you are done
cd ${TRAVIS_BUILD_DIR}