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

project (ECVL CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

#--------------------------------------------------------------------------------------------------------
# BEGIN: Check for c++17 support
#--------------------------------------------------------------------------------------------------------
# This should be done as follows but it doesn't work!?
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

# Removed to allow automatic tests without EDDLL
#--------------------------------------------------------------------------------------------------------
# BEGIN: add EDDLL dependency (still Windows specific)												
#--------------------------------------------------------------------------------------------------------
#set(ECVL_EDDLL_INSTALL_PATH "" CACHE PATH "")
#
#if(ECVL_EDDLL_INSTALL_PATH STREQUAL "")
#	message(ERROR " EDDLL install path must be specified")
#endif()
#
## TODO: handle the case in which EDDLL is dynamic compiled
#target_link_libraries(ECVL debug "${ECVL_EDDLL_INSTALL_PATH}/Debug/lib/static/EDDLL.lib")
##message(STATUS "${ECVL_EDDLL_INSTALL_PATH}/Debug/lib/static/EDDLL.lib")						 
#target_link_libraries(ECVL optimized "${ECVL_EDDLL_INSTALL_PATH}/Release/lib/static/EDDLL.lib")
##message(STATUS "${ECVL_EDDLL_INSTALL_PATH}/Release/lib/static/EDDLL.lib")
#include_directories("${ECVL_EDDLL_INSTALL_PATH}/include")
#--------------------------------------------------------------------------------------------------------
# END: add EDDLL dependency	(still Windows specific)											
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dinamic runtime 												
#--------------------------------------------------------------------------------------------------------
set(ECVL_STATIC ON CACHE BOOL "Whether to build static or dinamic libraries")

if(MSVC)
    if(ECVL_STATIC)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO) 
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else()
	    message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO) 
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
	endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: static/dinamic runtime										
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: OpenCV 	
#--------------------------------------------------------------------------------------------------------
set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc" CACHE STRING "OpenCV required packages")
FIND_PACKAGE(OpenCV REQUIRED ${OpenCV_REQUIRED_PACKAGES})
include_directories(${OpenCV_INCLUDE_DIRS})
#--------------------------------------------------------------------------------------------------------
# END: OpenCV 	
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: download and install GTest
#--------------------------------------------------------------------------------------------------------
set(CMAKE_FOLDER GTest)
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

include(GoogleTest)
set(CMAKE_FOLDER "")
#--------------------------------------------------------------------------------------------------------
# END: download and install GTest
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: ECVL library, examples and tests
#--------------------------------------------------------------------------------------------------------
add_executable(ECVL_TESTS "")
set(ECVL_MODULES "")

set(BUILD_ECVL_CORE ON CACHE BOOL "Whether to build ECVL_CORE or not")
set(BUILD_ECVL_GUI ON CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)")

if(BUILD_ECVL_CORE)
    add_library(ECVL_CORE "")
    add_subdirectory("modules/core")
    include_directories("modules/core/include")
    target_link_libraries(ECVL_CORE ${OpenCV_LIBS})
    set_target_properties(ECVL_CORE PROPERTIES FOLDER "Modules")
    list(APPEND ECVL_MODULES "ECVL_CORE")
endif()
if(BUILD_ECVL_GUI)
    if(BUILD_ECVL_CORE)
        find_package(wxWidgets COMPONENTS gl core base OPTIONAL_COMPONENTS net)
        if(wxWidgets_FOUND)
            include(${wxWidgets_USE_FILE})
            add_library(ECVL_GUI "")
            add_subdirectory("modules/gui")
            include_directories("modules/gui/include")
            #target_include_directories(ECVL_GUI PRIVATE ${wxWidgets_INCLUDE_DIRS})
            
            message(STATUS "wxWidgets include directories: ${wxWidgets_INCLUDE_DIRS}")   
            
            
            target_link_libraries(ECVL_GUI ECVL_CORE ${OpenCV_LIBS} ${wxWidgets_LIBRARIES})
            set_target_properties(ECVL_GUI PROPERTIES FOLDER "Modules")
            list(APPEND ECVL_MODULES "ECVL_GUI")
        else()
            set(BUILD_ECVL_GUI OFF CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)" FORCE)
            message(STATUS "ECVL_GUI disabled: unmet dependency wxWidgets")
        endif()
    else()
        set(BUILD_ECVL_GUI OFF CACHE BOOL "Whether to build ECVL_GUI or not (requires wxWidgets)" FORCE)
        message(STATUS "ECVL_GUI disabled: unmet dependency ECVL_CORE")
    endif()
endif()

target_link_libraries(ECVL_TESTS ${ECVL_MODULES} gtest_main)

message(STATUS "Modules to be built: ${ECVL_MODULES}")

enable_testing()

gtest_discover_tests(ECVL_TESTS)

add_executable(EXAMPLES "")
add_subdirectory(examples)
target_link_libraries(EXAMPLES ${ECVL_MODULES} ${wxWidgets_LIBRARIES})
#--------------------------------------------------------------------------------------------------------
# END: ECVL library, examples and tests
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: pthreads setup for windows os												
#--------------------------------------------------------------------------------------------------------
#
## The following commands should be removed when pthread linux 
## specific calls will be replaced by thread c++ standard calls 
#if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
#	set(ECVL_PTHREADS_INSTALL_PATH "$ENV{PTHREADS_ROOT}" CACHE PATH "")
#	
#	if(ECVL_PTHREADS_INSTALL_PATH STREQUAL "")
#		message(ERROR " pthreds install path must be specified under Windows")
#	endif()
#	
#	target_link_libraries(ECVL debug "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib"
#							   debug "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
#							   debug "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
#								 
#	target_link_libraries(ECVL optimized "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib"
#							   optimized "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
#							   optimized "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
#								 
#	include_directories("${ECVL_PTHREADS_INSTALL_PATH}/include")
#	
#	# The following command should be moved elsewhere
#	if(MSVC)
#		add_definitions(-D_CRT_SECURE_NO_WARNINGS)
#	endif()
#endif()
#--------------------------------------------------------------------------------------------------------
# END: pthreads setup for windows os												
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: pthreads setup for linux os												
#--------------------------------------------------------------------------------------------------------
#if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
#    find_package(Threads REQUIRED)
#    target_link_libraries(ECVL Threads::Threads)
#endif()
#--------------------------------------------------------------------------------------------------------
# END: pthreads setup for linux os												
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

