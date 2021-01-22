FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.10.0
)

if(ECVL_TESTS)
    if(ECVL_BUILD_DEPS)
        FetchContent_GetProperties(googletest)
        if(NOT googletest_POPULATED)
            FetchContent_Populate(googletest)
            SET(BUILD_GMOCK ON CACHE INTERNAL "" FORCE)
            SET(INSTALL_GTEST ON CACHE INTERNAL "" FORCE)
            if(ECVL_SHARED)
                set(BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
                set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
                target_compile_definitions(ECVL_TESTS PUBLIC GTEST_CREATE_SHARED_LIBRARY=1 GTEST_LINKED_AS_SHARED_LIBRARY=1)
            else()
                set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
                set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
            endif()
            add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
        else()
            find_package(GTest REQUIRED)
            find_package(GMock REQUIRED)
        endif()
    endif()
endif()