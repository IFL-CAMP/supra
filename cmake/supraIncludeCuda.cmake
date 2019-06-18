if(${CMAKE_VERSION} VERSION_LESS "3.10.0") 
    find_package(CUDA_old REQUIRED)
else()
	find_package(CUDA REQUIRED)
endif()

IF(CUDA_VERSION_MAJOR LESS 10)
	MESSAGE(FATAL_ERROR "CUDA >= 10.0 is required, but only found ${CUDA_VERSION}. Verify installed cuda toolkit and modify CUDA_TOOLKIT_ROOT_DIR if necessary.")
ENDIF()
