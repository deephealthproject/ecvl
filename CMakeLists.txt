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

project (ECVL)
add_library(ECVL "")

add_subdirectory(src)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#--------------------------------------------------------------------------------------------------------
# BEGIN: pthreads setup for windows os												
#--------------------------------------------------------------------------------------------------------

# The following commands should be removed when pthread linux 
# specific calls will be replaced by thread c++ standard calls 
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	set(ECVL_PTHREADS_INSTALL_PATH "" CACHE PATH "")
	
	if(ECVL_PTHREADS_INSTALL_PATH STREQUAL "")
		message(ERROR " pthreds install path must be specified under Windows")
	endif()
	
	target_link_libraries(ECVL debug "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib"
							   debug "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
							   debug "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
								 
	target_link_libraries(ECVL optimized "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib"
							   optimized "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
							   optimized "${ECVL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
								 
	include_directories("${ECVL_PTHREADS_INSTALL_PATH}/include")
	
	# The following command should be moved elsewhere
	if(MSVC)
		add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: pthreads setup for windows os												
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: pthreads setup for linux os												
#--------------------------------------------------------------------------------------------------------
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    find_package(Threads REQUIRED)
    target_link_libraries(EDDLL Threads::Threads)
endif()
#--------------------------------------------------------------------------------------------------------
# END: pthreads setup for linux os												
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# BEGIN: add EDDLL dependency (still Windows specific)												
#--------------------------------------------------------------------------------------------------------
set(ECVL_EDDLL_INSTALL_PATH "" CACHE PATH "")

if(ECVL_EDDLL_INSTALL_PATH STREQUAL "")
	message(ERROR " EDDL install path must be specified")
endif()

# TODO: handle the case in which EDDLL is dynamic compiled
target_link_libraries(ECVL debug "${ECVL_EDDLL_INSTALL_PATH}/Debug/lib/static/EDDLL.lib")
#message(STATUS "${ECVL_EDDLL_INSTALL_PATH}/Debug/lib/static/EDDLL.lib")						 
target_link_libraries(ECVL optimized "${ECVL_EDDLL_INSTALL_PATH}/Release/lib/static/EDDLL.lib")
#message(STATUS "${ECVL_EDDLL_INSTALL_PATH}/Release/lib/static/EDDLL.lib")
include_directories("${ECVL_EDDLL_INSTALL_PATH}/include")
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

target_link_libraries (ECVL  ${OpenCV_LIBS})
#--------------------------------------------------------------------------------------------------------
# END: OpenCV 	
#--------------------------------------------------------------------------------------------------------

add_executable(EXAMPLES "")
add_subdirectory(examples/)
target_link_libraries(EXAMPLES ECVL)