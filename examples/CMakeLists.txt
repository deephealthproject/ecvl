# ECVL - European Computer Vision Library
# Version: 0.1
# copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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
)
FOREACH(cur_file ${examples_sources})
    get_filename_component(exe_name ${cur_file} NAME_WE)
    set(cur_file "examples/${cur_file}")
    #message(STATUS "cur_file: ${cur_file}, exe_name: ${exe_name}")
    add_executable(${exe_name} ${cur_file})
    set_target_properties(${exe_name} PROPERTIES FOLDER "Examples")
    target_link_libraries(${exe_name} ${ECVL_MODULES})
ENDFOREACH()

if(ECVL_DATASET_PARSER)
    add_executable(example_dataset "examples/example_dataset.cpp")
    set_target_properties(example_dataset PROPERTIES FOLDER "Examples")
    target_link_libraries(example_dataset ${ECVL_MODULES})
    
    # Download and unpack MNIST dataset if necessary
    if (EXISTS "${CMAKE_BINARY_DIR}/mnist.zip")
        message(STATUS "${CMAKE_BINARY_DIR}/mnist.zip already exists, download skipped!")
    else()
        message(STATUS "Downloading MNIST...")
        file(DOWNLOAD http://imagelab.ing.unimore.it/files/DeepHealth/mnist.zip "${CMAKE_BINARY_DIR}/mnist.zip"
            STATUS status
            SHOW_PROGRESS
        )
        list( GET status 0 error_code )
        if( error_code )
            message(ERROR " Cmake is not able to download the mnist dataset. \nPlease, retry or download it manually (The dataset is also available at https://drive.google.com/file/d/1uWUkP3vMx1MWwOnZud7JhxcfmI2PZvv3/view?usp=sharing).")
            if(EXISTS "${CMAKE_BINARY_DIR}/mnist.zip")
                file(REMOVE "${CMAKE_BINARY_DIR}/mnist.zip")
            endif()
        else()
            message(STATUS "Downloading done.")
        endif()

        if(EXISTS "${CMAKE_BINARY_DIR}/mnist.zip")
            message(STATUS "Unpacking (it may take a while)...")
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_BINARY_DIR}/mnist.zip" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" OUTPUT_QUIET)
            message(STATUS "Unpacking done.")
        endif()
    endif()
endif()
if(ECVL_BUILD_EDDL)
    add_executable(example_ecvl_eddl "examples/example_ecvl_eddl.cpp")
    set_target_properties(example_ecvl_eddl PROPERTIES FOLDER "Examples")
    target_link_libraries(example_ecvl_eddl ${ECVL_MODULES})
endif()
if(ECVL_BUILD_GUI)
    add_executable(example_ecvl_gui "examples/example_ecvl_gui.cpp")
    set_target_properties(example_ecvl_gui PROPERTIES FOLDER "Examples")
    target_link_libraries(example_ecvl_gui ${ECVL_MODULES})
endif()
if(ECVL_WITH_DICOM)
    add_executable(example_nifti_dicom "examples/example_nifti_dicom.cpp")
    set_target_properties(example_nifti_dicom PROPERTIES FOLDER "Examples")
    target_link_libraries(example_nifti_dicom ${ECVL_MODULES})
endif()
if(ECVL_WITH_OPENSLIDE)
    add_executable(example_openslide "examples/example_openslide.cpp")
    set_target_properties(example_openslide PROPERTIES FOLDER "Examples")
    target_link_libraries(example_openslide ${ECVL_MODULES})
endif()

# Download and unpack ecvl examples data
if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data")
    message(STATUS "The data folder already exists, download skipped!")
else()
    message(STATUS "Downloading data files...")
    file(DOWNLOAD http://imagelab.ing.unimore.it/files/DeepHealth/ecvl_data.zip "${CMAKE_CURRENT_SOURCE_DIR}/ecvl_data.zip" STATUS status SHOW_PROGRESS)

    list(GET status 0 error_code)
    if(error_code)
        message("CMake is not able to download the ecvl examples data.")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ecvl_data.zip")
            file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/ecvl_data.zip")
        endif()
    else()
        message(STATUS "Downloading done.")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ecvl_data.zip")
        message(STATUS "Unpacking (it may take a while)...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_CURRENT_SOURCE_DIR}/ecvl_data.zip" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_QUIET)
        message(STATUS "Unpacking done.")
    endif()
endif()
