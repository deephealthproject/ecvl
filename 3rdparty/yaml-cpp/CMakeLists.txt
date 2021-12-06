FetchContent_Declare(
    yaml_cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
    GIT_TAG        yaml-cpp-0.6.3
)

if(ECVL_DATASET)
    if(ECVL_BUILD_DEPS)
        FetchContent_GetProperties(yaml_cpp)
        if(NOT yaml_cpp_POPULATED)
            FetchContent_Populate(yaml_cpp)
            # Add here options for yaml-cpp building
            if(ECVL_SHARED)
                SET(YAML_BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
                SET(YAML_MSVC_SHARED_RT ON CACHE INTERNAL "" FORCE)
            else()
                SET(YAML_BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
                SET(YAML_MSVC_SHARED_RT OFF CACHE INTERNAL "" FORCE)
            endif()
            SET(YAML_APPLE_UNIVERSAL_BIN OFF CACHE INTERNAL "" FORCE)
            SET(YAML_CPP_BUILD_CONTRIB OFF CACHE INTERNAL "" FORCE)
            SET(YAML_CPP_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
            SET(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "" FORCE)
            SET(YAML_CPP_INSTALL ON CACHE INTERNAL "" FORCE)
            
            mark_as_advanced(YAML_BUILD_SHARED_LIBS)
            mark_as_advanced(YAML_MSVC_STHREADED_RT)
            
            add_subdirectory(${yaml_cpp_SOURCE_DIR} ${yaml_cpp_BINARY_DIR})
        endif()
    else()
        find_package(yaml-cpp REQUIRED)
    endif()

    # Suppress yaml-cpp iterator warnings
    target_compile_definitions(yaml-cpp INTERFACE _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
    target_compile_options(yaml-cpp INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/wd4275 /wd4251>) # Suppress yaml-cpp warnings
    target_link_libraries(ECVL_CORE PUBLIC yaml-cpp)
endif()