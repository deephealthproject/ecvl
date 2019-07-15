cmake_minimum_required (VERSION 3.13)

project (nifti CXX)

add_library(nifti 

    include/nifti/nifti1.h
    include/nifti/nifti1_io.h
    include/nifti/znzlib.h
    nifti1_io.cpp
    znzlib.cpp

)

target_include_directories(nifti PUBLIC	
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
	$<INSTALL_INTERFACE:include>
)

#find_package(zlib)
#if (${ZLIB_FOUND})
#    target_link_libraries(nifti ZLIB::ZLIB)
#endif()

install(TARGETS nifti
        EXPORT NIFTIConfig
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib/static
		)
install(EXPORT NIFTIConfig DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
    DESTINATION "."
    FILES_MATCHING PATTERN "*.h")
