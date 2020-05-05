# ECVL - European Computer Vision Library
# Version: 0.2.1
# copyright (c) 2020, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
# Authors:
#    Costantino Grana (costantino.grana@unimore.it)
#    Federico Bolelli (federico.bolelli@unimore.it)
#    Michele Cancilla (michele.cancilla@unimore.it)
#    Laura Canalini (laura.canalini@unimore.it)
#    Stefano Allegretti (stefano.allegretti@unimore.it)
# All rights reserved.

cmake_minimum_required (VERSION 3.13)
#cmake_policy(SET CMP0091 NEW)  # must be set in order to use CMAKE_MSVC_RUNTIME_LIBRARY
project (ecvl CXX)

include(CheckCXXCompilerFlag)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")
set(CMAKE_POSITION_INDEPENDENT_CODE True) # To always generate position independent code

#--------------------------------------------------------------------------------------------------------
# BEGIN: Install(first step): must be at the beginning otherwise in case of any failure (i.e. OpenCV not found) won't work
#--------------------------------------------------------------------------------------------------------
# Set default install directories
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if (WIN32)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation Directory" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory" FORCE)
    endif()
endif()

if(CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
    SET(ECVL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ecvl)
else()
    SET(ECVL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
endif()
#--------------------------------------------------------------------------------------------------------
# END: Install (first step)
#--------------------------------------------------------------------------------------------------------

# Set build type (RELEASE/DEBUG)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#--------------------------------------------------------------------------------------------------------
# BEGIN: Check for c++17 support
#--------------------------------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
elseif(MSVC)
    check_cxx_compiler_flag("/std:c++17" COMPILER_SUPPORTS_CXX17)
else()
    # Unrecognized compiler
    set(COMPILER_SUPPORTS_CXX17 ON)
endif()

if(NOT COMPILER_SUPPORTS_CXX17)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

#--------------------------------------------------------------------------------------------------------
# END: Check for c++17 support
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: Check for CUDA support
#--------------------------------------------------------------------------------------------------------
set(ECVL_GPU_AVAILABLE OFF) # ON if GPU support is chosen AND actually available
option(ECVL_GPU "Build for GPU hardware (requires the NVIDIA CUDA toolkit)" ON)
if(ECVL_GPU)
    if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
        message(WARNING "Build for GPU hardware is only supported with CMake 3.17 or later.")
    else()
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            find_package(CUDAToolkit)
            if(CUDAToolkit_FOUND)
                set(ECVL_GPU_AVAILABLE ON)
                add_compile_definitions(ECVL_GPU)
            else()
                message(WARNING "The CUDA toolkit was not found, GPU support has been disabled.")
            endif()
        else()
            message(WARNING "No CUDA compiler was found, GPU support has been disabled.")
        endif()
    endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: Check for CUDA support
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dynamic runtime
#--------------------------------------------------------------------------------------------------------
set(ECVL_SHARED OFF CACHE BOOL "Whether to build static or dynamic libraries")
set(ecvl-type STATIC)

if (ECVL_SHARED)
    # TODO Shared building
    set(ecvl-type STATIC) #SHARED)
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

    # The following part should be made like this starting from CMake 3.15. It also requires policy CMP0091 set to NEW.
    # if(ECVL_SHARED)
        # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # else()
        # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # endif()

    if(ECVL_SHARED)
        message(STATUS "Setting dynamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
        if(ECVL_GPU_AVAILABLE)
            foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
                string(REGEX REPLACE "-MT" "-MD" ${flag_var} "${${flag_var}}")
            endforeach(flag_var)
        endif()
    else()
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
        if(ECVL_GPU_AVAILABLE)
            foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
                string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
            endforeach(flag_var)
        endif()
    endif()

endif()
#--------------------------------------------------------------------------------------------------------
# END: static/dynamic runtime
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL_CORE (the main target)
#--------------------------------------------------------------------------------------------------------
set(ECVL_BUILD_CORE ON CACHE BOOL "Whether to build ECVL_CORE or not" FORCE)
set(ECVL_TESTS ON CACHE BOOL "Whether to build ECVL tests or not")
if(ECVL_TESTS)
    add_executable(ECVL_TESTS "") # it must be defined before ECVL_CORE
endif()

add_library(ECVL_CORE ${ecvl-type})
add_subdirectory("modules/core")
target_include_directories(ECVL_CORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/core/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(ECVL_CORE PROPERTIES FOLDER "Modules")
set(ECVL_MODULES ECVL_CORE)
set(ECVL_MODULES_DIRS core)

if(ECVL_GPU_AVAILABLE)
    if(ECVL_SHARED)
        target_link_libraries(ECVL_CORE PUBLIC CUDA::cudart)
    else()
        target_link_libraries(ECVL_CORE PUBLIC CUDA::cudart_static)
    endif()
    # Enable cuda14 features
    target_compile_features(ECVL_CORE PUBLIC cuda_std_14)
endif()

# Enable c++17 features
target_compile_features(ECVL_CORE PUBLIC cxx_std_17)

# Enable link of stdc++fs library
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_link_libraries(ECVL_CORE PUBLIC stdc++fs)
endif()

#--------------------------------------------------------------------------------------------------------
# END: ECVL_CORE (the main target)
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: OpenCV
#--------------------------------------------------------------------------------------------------------
set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc;photo" CACHE STRING "OpenCV required packages")
find_package(OpenCV REQUIRED ${OpenCV_REQUIRED_PACKAGES})
target_include_directories(ECVL_CORE PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ECVL_CORE PUBLIC ${OpenCV_LIBS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
target_compile_definitions(ECVL_CORE PRIVATE OpenCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR})
#--------------------------------------------------------------------------------------------------------
# END: OpenCV
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: 3rdparty
#--------------------------------------------------------------------------------------------------------
set(ECVL_WITH_DICOM OFF CACHE BOOL "Whether to build DICOM support or not")
set(ECVL_WITH_OPENGL OFF CACHE BOOL "Whether to build OPENGL support or not (requires ECVL_GUI)")
set(ECVL_DATASET OFF CACHE BOOL "Whether to build yaml support or not")
set(ECVL_WITH_OPENSLIDE OFF CACHE BOOL "Whether to build OpenSlide support or not")
find_package(Git QUIET)
#--------------------------------------------------------------------------------------------------------
# END: 3rdparty
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL optional libraries
#--------------------------------------------------------------------------------------------------------
set(ECVL_BUILD_EXAMPLES OFF CACHE BOOL "Whether to build ECVL examples or not")
set(ECVL_BUILD_GUI OFF CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)")
set(ECVL_BUILD_EDDL ON CACHE BOOL "Whether to build ECVL_EDDL or not (requires eddl)")

if(ECVL_BUILD_GUI)
    if(ECVL_BUILD_CORE)
        find_package(wxWidgets COMPONENTS gl core base OPTIONAL_COMPONENTS net)
        if(wxWidgets_FOUND)
            include(${wxWidgets_USE_FILE})
            add_library(ECVL_GUI ${ecvl-type} "")
            add_subdirectory("modules/gui")
            include_directories("modules/gui/include")
            message(STATUS "wxWidgets include directories: ${wxWidgets_INCLUDE_DIRS}")

            # Remove these global defines, add to GUI only
            remove_definitions(-DUNICODE -D_UNICODE)
            target_compile_definitions(ECVL_GUI PRIVATE -DUNICODE -D_UNICODE)

            target_link_libraries(ECVL_GUI PUBLIC ECVL_CORE ${wxWidgets_LIBRARIES})
            target_include_directories(ECVL_GUI PUBLIC ${wxWidgets_INCLUDE_DIRS})
            set_target_properties(ECVL_GUI PROPERTIES FOLDER "Modules")
            list(APPEND ECVL_MODULES "ECVL_GUI")
            list(APPEND ECVL_MODULES_DIRS "gui")
        else()
            message(WARNING "Could NOT find wxWidgets")
        endif()
    else()
        set(ECVL_BUILD_GUI OFF CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)" FORCE)
        message(STATUS "ECVL_GUI disabled: unmet dependency ECVL_CORE")
    endif()
endif()

macro(F_BUILD_DATASET)
    add_library(DATASET ${ecvl-type} "modules/dataset/dataset.yaml")
    add_subdirectory("modules/dataset")
    include_directories("modules/dataset/include")
    target_link_libraries(DATASET ECVL_CORE)
    set_target_properties(DATASET PROPERTIES FOLDER "Modules")
    list(APPEND ECVL_MODULES "DATASET")
    list(APPEND ECVL_MODULES_DIRS "dataset")
endmacro()

if(ECVL_BUILD_EDDL)
    if(ECVL_BUILD_CORE)
        find_package(eddl)
        if(eddl_FOUND)
            set(ECVL_DATASET ON CACHE BOOL "Whether to build ECVL_DATASET or not" FORCE)
            F_BUILD_DATASET()
            add_library(ECVL_EDDL ${ecvl-type} "")
            add_subdirectory("modules/eddl")
            include_directories("modules/eddl/include")
            target_link_libraries(ECVL_EDDL DATASET eddl)
            set_target_properties(ECVL_EDDL PROPERTIES FOLDER "Modules")
            list(APPEND ECVL_MODULES "ECVL_EDDL")
            list(APPEND ECVL_MODULES_DIRS "eddl")
        endif()
    else()
        set(ECVL_BUILD_EDDL OFF CACHE BOOL "Whether to build ECVL_EDDL or not (requires eddl)" FORCE)
        message(STATUS "ECVL_EDDL disabled: unmet dependency ECVL_CORE")
    endif()
endif()
if(ECVL_DATASET AND NOT eddl_FOUND)
    if(ECVL_BUILD_CORE)
        F_BUILD_DATASET()
    else()
        set(ECVL_DATASET OFF CACHE BOOL "Whether to build ECVL_DATASET or not" FORCE)
        message(STATUS "ECVL_DATASET disabled: unmet dependency ECVL_CORE")
    endif()
endif()

foreach(MODULE ${ECVL_MODULES})
    string(TOLOWER ${MODULE} LOWER_MODULE)
    set_target_properties(${MODULE} PROPERTIES
        OUTPUT_NAME "${LOWER_MODULE}"
        DEBUG_POSTFIX d
        )
endforeach()

#--------------------------------------------------------------------------------------------------------
# END: ECVL optional libraries
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: 3rdparty
#--------------------------------------------------------------------------------------------------------
set(CMAKE_FOLDER 3rdparty)
add_subdirectory(3rdparty)
set(CMAKE_FOLDER "")

if(ECVL_WITH_DICOM)
    target_compile_definitions(ECVL_CORE PUBLIC ECVL_WITH_DICOM)
endif()

if(ECVL_WITH_OPENGL)
    target_compile_definitions(ECVL_GUI PUBLIC ECVL_WITH_OPENGL)
endif()

if(ECVL_WITH_OPENSLIDE)
    find_path(OPENSLIDE_INCLUDE_DIRECTORIES openslide.h PATH_SUFFIXES openslide DOC "Path to directory containing openslide.h")
    find_library(OPENSLIDE_LIBRARIES openslide DOC "Path to libopenslide.lib")
    if(OPENSLIDE_LIBRARIES AND OPENSLIDE_INCLUDE_DIRECTORIES)
        target_include_directories(ECVL_CORE PUBLIC ${OPENSLIDE_INCLUDE_DIRECTORIES})
        target_link_libraries(ECVL_CORE PUBLIC ${OPENSLIDE_LIBRARIES})
        target_compile_definitions(ECVL_CORE PUBLIC ECVL_WITH_OPENSLIDE)
        target_sources(ECVL_CORE PRIVATE "../modules/core/include/ecvl/core/support_openslide.h" "../modules/core/src/support_openslide.cpp")
    else()
        message(WARNING "Could NOT find OpenSlide")
    endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: 3rdparty
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: Examples
#--------------------------------------------------------------------------------------------------------
if (ECVL_BUILD_EXAMPLES)
    include(examples/CMakeLists.txt)
endif()
#--------------------------------------------------------------------------------------------------------
# END: Examples
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL tests
#--------------------------------------------------------------------------------------------------------
if(ECVL_TESTS)
    include(GoogleTest)
    enable_testing()
    target_link_libraries(ECVL_TESTS ${ECVL_MODULES} gtest gtest_main gmock)
    if(MSVC)
        target_compile_options(ECVL_TESTS PUBLIC "-bigobj")
    endif()
    if (ECVL_BUILD_EXAMPLES)
        target_compile_definitions(ECVL_TESTS PUBLIC ECVL_WITH_EXAMPLES)
    endif()
    gtest_discover_tests(ECVL_TESTS)
endif()
#--------------------------------------------------------------------------------------------------------
# END: ECVL tests
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: Install
#--------------------------------------------------------------------------------------------------------
install(TARGETS ${ECVL_MODULES}
        EXPORT ecvlTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
install(EXPORT ecvlTargets DESTINATION ${ECVL_CONFIG_INSTALL_PATH})
foreach(module_name ${ECVL_MODULES_DIRS})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/modules/${module_name}/include
        DESTINATION "."
        FILES_MATCHING PATTERN "*.h")
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ecvlConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/ecvlConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cmake/ecvlConfig.cmake
        DESTINATION ${ECVL_CONFIG_INSTALL_PATH})
#--------------------------------------------------------------------------------------------------------
# END: Install
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: Summary
#--------------------------------------------------------------------------------------------------------
message(STATUS "--------------------------------------")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE} - ${ecvl-type}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
# message(STATUS "ECVL MODULES DIRS ${ECVL_MODULES_DIRS}")
message(STATUS "ECVL modules to build: ${ECVL_MODULES}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
#--------------------------------------------------------------------------------------------------------
# END: Summary
#--------------------------------------------------------------------------------------------------------