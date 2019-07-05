cmake_minimum_required (VERSION 3.13)

project (nifti CXX)

add_library(nifti 

    include/nifti/nifti1.h
    include/nifti/nifti1_io.h
    include/nifti/znzlib.h
    nifti1_io.cpp
    znzlib.cpp

)

target_include_directories(nifti PUBLIC include)

find_package(zlib)
if (${ZLIB_FOUND})
    target_link_libraries(nifti ZLIB::ZLIB)
endif()



