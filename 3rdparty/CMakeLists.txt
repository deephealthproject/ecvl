macro(ecvl_update_3rdparty SUBMODULE)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodule as needed
        message(STATUS "${SUBMODULE} update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive "${SUBMODULE}"
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout ${SUBMODULE} submodule")
        endif()
    endif()
endmacro()

if(${ECVL_WITH_OPENGL})
    if(${ECVL_BUILD_GUI})
        find_package(OpenGL)
        if(${OPENGL_FOUND})
            target_link_libraries(ECVL_GUI PUBLIC OpenGL::GL)
            target_include_directories(ECVL_GUI PUBLIC ${OPENGL_INCLUDE_DIR})

            ecvl_update_3rdparty(glm)

            if (ECVL_SHARED)
                SET(BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
                SET(BUILD_STATIC_LIBS OFF CACHE INTERNAL "" FORCE)
            else()
                SET(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
                SET(BUILD_STATIC_LIBS ON CACHE INTERNAL "" FORCE)
            endif()
            SET(GLM_DISABLE_AUTO_DETECTION OFF CACHE INTERNAL "" FORCE)
            SET(GLM_QUIET OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_CXX_11 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_CXX_14 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_CXX_17 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_CXX_20 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_CXX_98 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_FAST_MATH OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_LANG_EXTENSIONS OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_AVX OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_AVX2 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_SSE2 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_SSE3 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_SSE4_1 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_SSE4_2 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_ENABLE_SIMD_SSSE3 OFF CACHE INTERNAL "" FORCE)
            SET(GLM_TEST_FORCE_PURE OFF CACHE INTERNAL "" FORCE)

            set(CMAKE_FOLDER 3rdparty/glm)
            add_subdirectory(glm)
            set(CMAKE_FOLDER 3rdparty)
            target_link_libraries(ECVL_GUI PUBLIC glm)

            #target_include_directories(ECVL_GUI PUBLIC glad/include)

            target_include_directories(ECVL_GUI PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glad/include>
                $<INSTALL_INTERFACE:include>
            )
            install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad/include
                DESTINATION "."
                FILES_MATCHING PATTERN "*.h")
            # should the header files be installed separately?

            target_sources(ECVL_GUI PRIVATE glad/src/glad.cpp)
        else()
            set(ECVL_WITH_OPENGL OFF CACHE BOOL "Whether to build OPENGL support or not (requires ECVL_GUI)" FORCE)
            message(STATUS "ECVL_WITH_OPENGL disabled: OpenGL not found")
        endif()
    else(${ECVL_BUILD_GUI})
        set(ECVL_WITH_OPENGL OFF CACHE BOOL "Whether to build OPENGL support or not (requires ECVL_GUI)" FORCE)
        message(STATUS "ECVL_WITH_OPENGL disabled: unmet dependency ECVL_GUI")
    endif(${ECVL_BUILD_GUI})
endif(${ECVL_WITH_OPENGL})

# nifti
# we use our own functions
#set(CMAKE_FOLDER 3rdparty/nifti)
#add_subdirectory(nifti)
#target_link_libraries(ECVL_CORE PUBLIC nifti)
#set(CMAKE_FOLDER 3rdparty)

# prova externalproject

#include(ExternalProject)
#
#ExternalProject_Add(nifti
#  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nifti
#  BINARY_DIR ${CMAKE_BINARY_DIR}/nifti
#  DOWNLOAD_COMMAND ""
#  UPDATE_COMMAND ""
#  CMAKE_CACHE_ARGS
#    #-Foo_DIR:PATH=${CMAKE_BINARY_DIR}/Foo-build
#    #-DAwesome_ENABLE_EXTRA:BOOL=${Awesome_ENABLE_EXTRA}
#    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}/3rdparty
#  #INSTALL_COMMAND ""
#  )
##return()
#
##add_subdirectory(nifti)
#
##find_package(NIFTI REQUIRED)
#add_library(nifti IMPORTED STATIC)
#target_link_libraries(ECVL_CORE PUBLIC nifti)


if(${ECVL_TESTS})

    ecvl_update_3rdparty(googletest)

    SET(BUILD_GMOCK OFF CACHE INTERNAL "" FORCE)

    set(CMAKE_FOLDER 3rdparty/googletest)
    add_subdirectory(googletest)
    set(CMAKE_FOLDER 3rdparty)

    include(GoogleTest)

endif(${ECVL_TESTS})

# --------------------
# yaml-cpp
# --------------------
if (${ECVL_DATASET_PARSER})
    ecvl_update_3rdparty(yaml-cpp)

    SET(APPLE_UNIVERSAL_BIN OFF CACHE INTERNAL "" FORCE)
    SET(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
    SET(YAML_MSVC_SHARED_RT OFF CACHE INTERNAL "" FORCE)
    #SET(MSVC_STHREADED_RT OFF CACHE INTERNAL "" FORCE)
    SET(YAML_CPP_BUILD_CONTRIB OFF CACHE INTERNAL "" FORCE)
    SET(YAML_CPP_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
    SET(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "" FORCE)
    SET(YAML_CPP_INSTALL ON CACHE INTERNAL "" FORCE)
    SET(BUILD_TESTING OFF CACHE INTERNAL "" FORCE)

    set(CMAKE_FOLDER 3rdparty/yaml-cpp)
    add_subdirectory(yaml-cpp)
    set(CMAKE_FOLDER 3rdparty)
    target_link_libraries(ECVL_CORE PUBLIC yaml-cpp)
    message(STATUS "yaml-cpp added")
endif()