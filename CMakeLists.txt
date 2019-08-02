# This file is part of ECVL an European Computer Vision Library.
# Developed within the DeepHealth project.
# Boosting AI in Europe.
#
# The MIT License (MIT)
#
# Copyright (c) 2019
#           Federico Bolelli, <federico.bolelli@unimore.it>
#           Costantino Grana, <costantino.grana@unimore.it>
#			Stefano Allegretti, <stefano.allegretti@unimore.it>
#			Laura Canalini, <laura.canalini@unimore.it>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required (VERSION 3.13)

project (ecvl CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")
set(CMAKE_POSITION_INDEPENDENT_CODE True) #To always generate position independent code

#--------------------------------------------------------------------------------------------------------
# BEGIN: Install(first step): must be at the beginning otherwise in case of any failure (i.e. OpenCV not found) won't work
#--------------------------------------------------------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "\nHere we are!\n")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
endif()
#--------------------------------------------------------------------------------------------------------
# END: Install (first step)
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: Check for c++17 support
#--------------------------------------------------------------------------------------------------------
# This should be done as follows but it doesn't work!?
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
    
    if(COMPILER_SUPPORTS_CXX17 AND NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+17")
        set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}" CACHE STRING "Enable c++17" FORCE)
    elseif(NOT COMPILER_SUPPORTS_CXX17)
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
    endif()
else()
    if(MSVC)
        if(MSVC_VERSION LESS 1914)
            message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
        endif()
    else()
        message(STATUS "We are not able to check if ${CMAKE_CXX_COMPILER} compiler has C++17 support. ECVL may not work properly.")
    endif()
endif()
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
#--------------------------------------------------------------------------------------------------------
# END: Check for c++17 support
#--------------------------------------------------------------------------------------------------------

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dinamic runtime 												
#--------------------------------------------------------------------------------------------------------
set(ECVL_SHARED OFF CACHE BOOL "Whether to build static or dinamic libraries")

if(MSVC)
    if(ECVL_SHARED)
    	message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO) 
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else()
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO ) 
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
	endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: static/dinamic runtime										
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL_CORE (the main target)	
#--------------------------------------------------------------------------------------------------------
set(ECVL_TESTS ON CACHE BOOL "Whether to build ECVL tests or not")
if(${ECVL_TESTS})
    add_executable(ECVL_TESTS "") # it must be defined before ECVL_CORE
endif()

add_library(ECVL_CORE STATIC "")
add_subdirectory("modules/core")
target_include_directories(ECVL_CORE PUBLIC	
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/core/include>  
	$<INSTALL_INTERFACE:include>
)
set_target_properties(ECVL_CORE PROPERTIES FOLDER "Modules")
set(ECVL_MODULES ECVL_CORE)
set(ECVL_MODULES_DIRS core)
#--------------------------------------------------------------------------------------------------------
# END: ECVL_CORE (the main target)	
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: OpenCV 	
#--------------------------------------------------------------------------------------------------------
set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc" CACHE STRING "OpenCV required packages")
FIND_PACKAGE(OpenCV REQUIRED ${OpenCV_REQUIRED_PACKAGES})
target_include_directories(ECVL_CORE PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ECVL_CORE PUBLIC ${OpenCV_LIBS})
#--------------------------------------------------------------------------------------------------------
# END: OpenCV 	
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: 3rdparty
#--------------------------------------------------------------------------------------------------------
set(ECVL_WITH_DICOM OFF CACHE BOOL "Whether to build DICOM support or not")
set(ECVL_WITH_OPENGL OFF CACHE BOOL "Whether to build OPENGL support or not (requires ECVL_GUI)")

find_package(Git QUIET)
#--------------------------------------------------------------------------------------------------------
# END: 3rdparty
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: download and install GTest
#--------------------------------------------------------------------------------------------------------
#set(CMAKE_FOLDER GTest)
#configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
#execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#  RESULT_VARIABLE result
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
#if(result)
#  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
#endif()
#execute_process(COMMAND ${CMAKE_COMMAND} --build .
#  RESULT_VARIABLE result
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
#if(result)
#  message(FATAL_ERROR "Build step for googletest failed: ${result}")
#endif()
#
## Prevent overriding the parent project's compiler/linker
## settings on Windows
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#
## Add googletest directly to our build. This defines
## the gtest and gtest_main targets.
#add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
#                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
#                 EXCLUDE_FROM_ALL)
#
## The gtest/gtest_main targets carry header search path
## dependencies automatically when using CMake 2.8.11 or
## later. Otherwise we have to add them here ourselves.
#if (CMAKE_VERSION VERSION_LESS 2.8.11)
#  include_directories("${gtest_SOURCE_DIR}/include")
#endif()
#
#include(GoogleTest)
#set(CMAKE_FOLDER "")
#--------------------------------------------------------------------------------------------------------
# END: download and install GTest
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL optional libraries
#--------------------------------------------------------------------------------------------------------
set(ECVL_BUILD_CORE ON CACHE BOOL "Whether to build ECVL_CORE or not" FORCE)
set(ECVL_BUILD_GUI ON CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)")
set(ECVL_BUILD_EDDL ON CACHE BOOL "Whether to build ECVL_EDDL or not (requires EDDL)")

if(ECVL_BUILD_GUI)
    if(ECVL_BUILD_CORE)
        find_package(wxWidgets COMPONENTS gl core base OPTIONAL_COMPONENTS net)
        if(wxWidgets_FOUND)
            include(${wxWidgets_USE_FILE})
            add_library(ECVL_GUI STATIC "")
            add_subdirectory("modules/gui")
            include_directories("modules/gui/include")
            
            message(STATUS "wxWidgets include directories: ${wxWidgets_INCLUDE_DIRS}")   
            
            target_link_libraries(ECVL_GUI PUBLIC ECVL_CORE ${wxWidgets_LIBRARIES})
            target_include_directories(ECVL_GUI PUBLIC ${wxWidgets_INCLUDE_DIRS})

            set_target_properties(ECVL_GUI PROPERTIES FOLDER "Modules")
            list(APPEND ECVL_MODULES "ECVL_GUI")
            list(APPEND ECVL_MODULES_DIRS "gui")
        else()
            set(ECVL_BUILD_GUI OFF CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)" FORCE)
            message(STATUS "ECVL_GUI disabled: unmet dependency wxWidgets")
        endif()
    else()
        set(ECVL_BUILD_GUI OFF CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)" FORCE)
        message(STATUS "ECVL_GUI disabled: unmet dependency ECVL_CORE")
    endif()
endif()
if(ECVL_BUILD_EDDL)
    if(ECVL_BUILD_CORE)
        find_package(EDDL)
        if(EDDL_FOUND)
            add_library(ECVL_EDDL STATIC "")
            add_subdirectory("modules/eddl")
            include_directories("modules/eddl/include")
           
            target_link_libraries(ECVL_EDDL ECVL_CORE eddl)
            set_target_properties(ECVL_EDDL PROPERTIES FOLDER "Modules")
            list(APPEND ECVL_MODULES "ECVL_EDDL")
            list(APPEND ECVL_MODULES_DIRS "eddl")
        else()
            set(ECVL_BUILD_EDDL OFF CACHE BOOL "Whether to build ECVL_EDDL or not (requires EDDL)" FORCE)
            message(STATUS "ECVL_EDDL disabled: unmet dependency EDDL")
        endif()
    else()
        set(ECVL_BUILD_EDDL OFF CACHE BOOL "Whether to build ECVL_EDDL or not (requires EDDL)" FORCE)
        message(STATUS "ECVL_EDDL disabled: unmet dependency ECVL_CORE")
    endif()
endif()

foreach(MODULE ${ECVL_MODULES})
    string(TOLOWER ${MODULE} LOWER_MODULE)
	set_target_properties(${MODULE} PROPERTIES OUTPUT_NAME_DEBUG "${LOWER_MODULE}d")
	set_target_properties(${MODULE} PROPERTIES OUTPUT_NAME_RELEASE "${LOWER_MODULE}")
endforeach()

message(STATUS "Modules to be built: ${ECVL_MODULES}")
#--------------------------------------------------------------------------------------------------------
# END: ECVL optional libraries
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: 3rdparty
#--------------------------------------------------------------------------------------------------------
set(CMAKE_FOLDER 3rdparty)
add_subdirectory(3rdparty)
set(CMAKE_FOLDER "")

if(${ECVL_WITH_DICOM})
    add_compile_definitions(ECVL_WITH_DICOM)
endif(${ECVL_WITH_DICOM})

if(${ECVL_WITH_OPENGL})
    add_compile_definitions(ECVL_WITH_OPENGL)
endif(${ECVL_WITH_OPENGL})
#--------------------------------------------------------------------------------------------------------
# END: 3rdparty
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL examples and tests
#--------------------------------------------------------------------------------------------------------
if (${ECVL_TESTS})
    target_link_libraries(ECVL_TESTS ${ECVL_MODULES} gtest gtest_main)
    enable_testing()
    #gtest_discover_tests(ECVL_TESTS)
    gtest_add_tests(TARGET ECVL_TESTS)
endif()

add_executable(EXAMPLES "")
add_subdirectory(examples)
target_link_libraries(EXAMPLES ${ECVL_MODULES})
#--------------------------------------------------------------------------------------------------------
# END: ECVL examples and tests
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: Snippets
#--------------------------------------------------------------------------------------------------------
include(snippets/CMakeLists.txt)
FOREACH(cur_file ${snippets_sources})
    get_filename_component(exe_name ${cur_file} NAME_WE)
    set(cur_file "snippets/${cur_file}")
    #message(STATUS "cur_file: ${cur_file}, exe_name: ${exe_name}")
    add_executable(${exe_name} ${cur_file})
	set_target_properties(${exe_name} PROPERTIES FOLDER "Snippets")
    target_link_libraries(${exe_name} ${ECVL_MODULES})
ENDFOREACH()
#--------------------------------------------------------------------------------------------------------
# END: Snippets
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: Install
#--------------------------------------------------------------------------------------------------------
install(TARGETS ${ECVL_MODULES}
		EXPORT ECVLTargets
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
install(EXPORT ECVLTargets DESTINATION ${CMAKE_INSTALL_PREFIX})
foreach(module_name ${ECVL_MODULES_DIRS})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/modules/${module_name}/include
		DESTINATION "."
        FILES_MATCHING PATTERN "*.h")
endforeach()
message(STATUS "ECVL MODULES DIRS ${ECVL_MODULES_DIRS}")   

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ECVLConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/ECVLConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cmake/ECVLConfig.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}) 
#--------------------------------------------------------------------------------------------------------
# END: Install
#--------------------------------------------------------------------------------------------------------
        