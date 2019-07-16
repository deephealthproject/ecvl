target_sources(ECVL_CORE
	PRIVATE
    
    core.h
	core/arithmetic.h
	core/datatype.h
	core/datatype_existing_tuples_inc.h
    core/datatype_existing_tuples_signed_inc.h
    core/datatype_existing_tuples_unsigned_inc.h
	core/datatype_matrix.h
	core/datatype_tuples_inc.h
	core/filesystem.h
	core/image.h
	core/imgcodecs.h
	core/imgproc.h
	core/iterators.h
	core/iterators_impl.inc.h
	core/memorymanager.h
    core/standard_errors.h
	core/support_opencv.h
	core/support_nifti.h
    core/type_promotion.h

)