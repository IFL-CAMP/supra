CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )

# add a target to generate API documentation with Doxygen
IF(SUPRA_BUILD_DOC)
	find_package(Doxygen REQUIRED)
	find_package(PythonInterp REQUIRED)
ELSE()
	find_package(Doxygen QUIET)
	find_package(PythonInterp QUIET)
ENDIF()
if(${DOXYGEN_FOUND} AND ${PYTHONINTERP_FOUND})
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	if(SUPRA_BUILD_DOC)
		add_custom_target(DOXYGEN ALL
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM)
	ELSE()
		add_custom_target(DOXYGEN
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM)
	ENDIF()
	
	
	add_custom_target(DOXYGEN_PARAMETERS
		${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/searchParameters.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../tools
		COMMENT "Generating the parameter description for doxygen" VERBATIM)
	
	add_dependencies(DOXYGEN DOXYGEN_PARAMETERS)	
endif()