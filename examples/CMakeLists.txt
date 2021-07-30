# ECVL - European Computer Vision Library
# Version: 0.3.4
# copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
# Authors:
#    Costantino Grana (costantino.grana@unimore.it)
#    Federico Bolelli (federico.bolelli@unimore.it)
#    Michele Cancilla (michele.cancilla@unimore.it)
#    Laura Canalini (laura.canalini@unimore.it)
#    Stefano Allegretti (stefano.allegretti@unimore.it)
# All rights reserved.

set(examples_sources
    example_core_iterators.cpp
    example_imagewatch.cpp
    example_image_view.cpp
    example_imgcodecs.cpp
    example_imgproc.cpp
    example_moments.cpp
    example_performance.cpp
	example_threshold.cpp
)

if(ECVL_DATASET)
    list(APPEND examples_sources example_dataset_parser.cpp example_dataset_generator.cpp)
endif()
if(ECVL_BUILD_EDDL AND eddl_FOUND)
    list(APPEND examples_sources example_ecvl_eddl.cpp example_pipeline.cpp example_pipeline_UC12_omp.cpp)
endif()
if(ECVL_BUILD_GUI AND wxWidgets_FOUND)
    list(APPEND examples_sources example_ecvl_gui.cpp)
endif()
if(ECVL_WITH_DICOM)
    list(APPEND examples_sources example_nifti_dicom.cpp)
endif()
if(ECVL_WITH_OPENSLIDE AND OPENSLIDE_LIBRARIES AND OPENSLIDE_INCLUDE_DIRECTORIES)
    list(APPEND examples_sources example_openslide.cpp)
endif()

FOREACH(cur_file ${examples_sources})
    get_filename_component(exe_name ${cur_file} NAME_WE)
    set(cur_file "examples/${cur_file}")
    #message(STATUS "cur_file: ${cur_file}, exe_name: ${exe_name}")
    add_executable(${exe_name} ${cur_file})
    set_target_properties(${exe_name} PROPERTIES FOLDER "Examples")
    target_link_libraries(${exe_name} ${ECVL_MODULES})
ENDFOREACH()

# Download and unpack ecvl examples data
if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/data")
    message(STATUS "The data folder already exists, download skipped!")
else()
    message(STATUS "Downloading examples data...")
    file(DOWNLOAD https://www.dropbox.com/s/fe3bo0206eklofh/data.zip "${CMAKE_CURRENT_SOURCE_DIR}/examples/data.zip" STATUS status SHOW_PROGRESS)
    list(GET status 0 error_code)
    if(error_code)
        message("CMake is not able to download the ecvl examples data.")
    else()
        message(STATUS "Downloading done.")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/data.zip")
            message(STATUS "Unpacking (it may take a while)...")
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf "${CMAKE_CURRENT_SOURCE_DIR}/examples/data.zip" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples)
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf "${CMAKE_CURRENT_SOURCE_DIR}/examples/data/mnist.zip" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/data)
            message(STATUS "Unpacking done.")
        endif()
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/data.zip")
        file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/examples/data.zip")
        file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/examples/data/mnist.zip")
    endif()
endif()

configure_file("${CMAKE_SOURCE_DIR}/modules/dataset/test/dataset_path.h.cmake" "${CMAKE_SOURCE_DIR}/modules/dataset/test/dataset_path.h")
