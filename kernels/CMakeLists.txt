set(ECVL_KERNELS_SRC
	#    "${CMAKE_CURRENT_SOURCE_DIR}/resize_accel.cpp"
	#  "${CMAKE_CURRENT_SOURCE_DIR}/threshold_accel.cpp"
	#  "${CMAKE_CURRENT_SOURCE_DIR}/otsuThreshold_accel.cpp"
	#  "${CMAKE_CURRENT_SOURCE_DIR}/gaussian_accel.cpp"
	#    "${CMAKE_CURRENT_SOURCE_DIR}/rgb2gray_accel.cpp"
		# "${CMAKE_CURRENT_SOURCE_DIR}/gray2rgb_accel.cpp"
	    # "${CMAKE_CURRENT_SOURCE_DIR}/flipvertical_accel.cpp"
	    # "${CMAKE_CURRENT_SOURCE_DIR}/mirror_accel.cpp"
		 "${CMAKE_CURRENT_SOURCE_DIR}/min_max_loc_accel.cpp"
	     "${CMAKE_CURRENT_SOURCE_DIR}/histogram_accel.cpp"
	    "${CMAKE_CURRENT_SOURCE_DIR}/integral_image_accel.cpp"
	 	  "${CMAKE_CURRENT_SOURCE_DIR}/gamma_correction_accel.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/filter2d_accel.cpp"
	PARENT_SCOPE
)
