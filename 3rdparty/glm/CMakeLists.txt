FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        f920ffabb21798fb35a7a71e939acfc1933faadd
)

if(ECVL_WITH_OPENGL)
    if(ECVL_BUILD_GUI AND wxWidgets_FOUND)
        find_package(OpenGL)
        if(OPENGL_FOUND)
            target_link_libraries(ECVL_GUI PUBLIC OpenGL::GL)
            target_include_directories(ECVL_GUI PUBLIC ${OPENGL_INCLUDE_DIR})

            if(ECVL_BUILD_DEPS)
                FetchContent_GetProperties(glm)
                if(NOT glm_POPULATED)
                    FetchContent_Populate(glm)
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

                    add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
                endif()
            else()
                find_package(glm REQUIRED)
            endif()
            target_link_libraries(ECVL_GUI PUBLIC glm)

            target_include_directories(ECVL_GUI PUBLIC
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/3rdparty/glad/include>
                $<INSTALL_INTERFACE:include>
            )
            install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/glad/include
                DESTINATION "."
                FILES_MATCHING PATTERN "*.h")
            # should the header files be installed separately?

            target_sources(ECVL_GUI PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/glad/src/glad.cpp)
        else()
            set(ECVL_WITH_OPENGL OFF CACHE BOOL "Whether to build OPENGL support or not (requires ECVL_GUI)" FORCE)
            message(STATUS "ECVL_WITH_OPENGL disabled: OpenGL not found")
        endif()
    else()
        set(ECVL_WITH_OPENGL OFF CACHE BOOL "Whether to build OPENGL support or not (requires ECVL_GUI)" FORCE)
        message(STATUS "ECVL_WITH_OPENGL disabled: unmet dependency ECVL_GUI")
    endif()
endif()
