set(examples_sources
	example_imgproc.cpp
	example_image_view.cpp
)

if(ECVL_BUILD_EDDL)
	add_executable(example_ecvl_eddl "examples/example_ecvl_eddl.cpp")
	set_target_properties(example_ecvl_eddl PROPERTIES FOLDER "Examples")
    target_link_libraries(example_ecvl_eddl ${ECVL_MODULES})
	
	# Download and unpack MNIST dataset if necessary
	if (EXISTS "${CMAKE_BINARY_DIR}/mnist.zip")
		message(STATUS "${CMAKE_BINARY_DIR}/mnist.zip already exists, download skipped!")
	else()
		message(STATUS "Downloading MNIST...")
		file(DOWNLOAD http://imagelab.ing.unimore.it/files/DeepHealth/mnist.zip "${CMAKE_BINARY_DIR}/mnist.zip"
			TIMEOUT 60
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