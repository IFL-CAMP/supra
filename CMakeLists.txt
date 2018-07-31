CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA")
PROJECT(nvSUPRA)

MESSAGE( STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
MESSAGE( STATUS "CMAKE_CL_64: ${CMAKE_CL_64}")

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
IF(WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
ENDIF(WIN32)

# options for modules to include
OPTION(SUPRA_PROFILING             "Add profiling message to logfile"  ON)
OPTION(SUPRA_CUDA                  "Use cuda in SUPRA"                 ON)
OPTION(SUPRA_BEAMFORMER            "Build Software beamformer"         ON)
OPTION(SUPRA_DEVICE_ITK_FILE_OUT   "Build ITK file output Interface"   ON)

IF(NOT SUPRA_CUDA)
	MESSAGE(WARNING "CUDA has been disabled. This will exclude a number of nodes from the build.")
	IF(SUPRA_BEAMFORMER)
		MESSAGE(WARNING "The software beamformer requires CUDA, I disabled it.")
		SET(SUPRA_BEAMFORMER OFF CACHE BOOL "Build Software beamformer" FORCE)
	ENDIF()
ENDIF()

# external folder
SET( SUPRA_EXTERNALS "${CMAKE_CURRENT_SOURCE_DIR}/external" CACHE PATH "Path to external libs e.g. ultrasonix.")
SET( SUPRA_EXTERNALS_TBB      ""  CACHE PATH "Path to the TBB base dir")
SET( SUPRA_EXTERNALS_TBB_LIBS ""  CACHE PATH "Path to the TBB lib dir")
	
ADD_SUBDIRECTORY(src/SupraLib)
ADD_SUBDIRECTORY(src/Wrapper)
