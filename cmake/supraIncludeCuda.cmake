find_package(CUDA REQUIRED)
IF(CUDA_VERSION_MAJOR LESS 10)
	MESSAGE(FATAL_ERROR "CUDA >= 10.0 is required, but only found ${CUDA_VERSION}. Verify installed cuda toolkit and modify CUDA_TOOLKIT_ROOT_DIR if necessary.")
ENDIF()
