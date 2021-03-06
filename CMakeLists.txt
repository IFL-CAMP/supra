CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA")
PROJECT(SUPRA)

SET(MAJOR_VERSION "1")
SET(MINOR_VERSION "0")
SET(PATCH_VERSION "0")

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
OPTION(SUPRA_PROFILING             "Add profiling message to logfile"  OFF)
OPTION(SUPRA_WARNINGS_AS_ERRORS    "Treat all build warnings as errors (NOT on windows)" OFF)
OPTION(SUPRA_DOWNLOAD_SAMPLEDATA   "Whether sample data should be downloaded" ON)
OPTION(SUPRA_CUDA                  "Use cuda in SUPRA"                 ON)
OPTION(SUPRA_CUDA_PORTABLE         "Build for all supported CUDA architectures" OFF)
OPTION(SUPRA_TORCH                 "Use pytorch in SUPRA"              OFF)
OPTION(SUPRA_BUILD_DOC             "Build doxygen documentation"       OFF)
OPTION(SUPRA_INTERFACE_GRAPHIC     "Build Graphic Interface (QT)"      ON)
OPTION(SUPRA_INTERFACE_GRAPHIC_CMD "Activate Console Output for GUI"   ON)
OPTION(SUPRA_INTERFACE_COMMANDLINE "Build Commandline Interface"       ON)
OPTION(SUPRA_INTERFACE_EXECUTOR    "Build Supra Executor"              ON)
OPTION(SUPRA_INTERFACE_ROS         "Build ROS Interface"               OFF)
OPTION(SUPRA_INTERFACE_REST        "Build REST Interface"              OFF)
OPTION(SUPRA_BEAMFORMER            "Build Software beamformer"         ON)
OPTION(SUPRA_DEVICE_ULTRASOUND_SIM "Build Ultrasound Test"             ON)
OPTION(SUPRA_DEVICE_TRACKING_SIM   "Build Tracker Test"                ON)
OPTION(SUPRA_DEVICE_TRACKING_IGTL  "Build OpenIGTLink Tracking Input"  ON)
OPTION(SUPRA_DEVICE_ULTRASONIX     "Build Ultrasonix Ultrasound"       OFF)
OPTION(SUPRA_DEVICE_IGTL_OUTPUT    "Build OpenIGTLink Output"          ON)
OPTION(SUPRA_DEVICE_CEPHASONICS    "Build Cephasonics Ultrasound"      OFF)
OPTION(SUPRA_DEVICE_ITK_FILE_OUT   "Build ITK file output Interface"   ON)
OPTION(SUPRA_DEVICE_ROS_IMAGE_OUT  "Build ROS image output Interface"  OFF)
OPTION(SUPRA_DEVICE_ROS_EDEN2020   "Build EDEN2020 ROS image output Interface"  OFF)
SET( SUPRA_DEVICE_ULTRASONIX_V     "5.7"  CACHE STRING "Ultrasonix Version (5.7|6.07)")
IF(WIN32)
	# Ros tracking output is by default activated on windows as it does not add dependencies
	OPTION(SUPRA_DEVICE_TRACKING_ROS   "Build ROS Tracking Input"          ON)
ELSE()
	OPTION(SUPRA_DEVICE_TRACKING_ROS   "Build ROS Tracking Input"          OFF)
ENDIF()

IF(NOT SUPRA_CUDA)
	MESSAGE(WARNING "CUDA has been disabled. This will exclude a number of nodes from the build.")
	IF(SUPRA_BEAMFORMER)
		MESSAGE(WARNING "The software beamformer requires CUDA, I disabled it.")
		SET(SUPRA_BEAMFORMER OFF CACHE BOOL "Build Software beamformer" FORCE)
	ENDIF()
	IF(SUPRA_DEVICE_CEPHASONICS)
		MESSAGE(WARNING "Without CUDA, only the built-in Cephasonics pipeline can be used, but not channel capture")
	ENDIF()
ENDIF()

# external folder
SET( SUPRA_EXTERNALS "${CMAKE_CURRENT_SOURCE_DIR}/external" CACHE PATH "Path to external libs e.g. ultrasonix.")
SET( SUPRA_EXTERNALS_TBB      ""  CACHE PATH "Path to the TBB base dir")
SET( SUPRA_EXTERNALS_TBB_LIBS ""  CACHE PATH "Path to the TBB lib dir")

IF(SUPRA_TORCH)
	find_package(Torch REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
ENDIF(SUPRA_TORCH)

IF(SUPRA_WARNINGS_AS_ERRORS)
    IF(CMAKE_COMPILER_IS_GNUCXX)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDIF(SUPRA_WARNINGS_AS_ERRORS)
	
ADD_SUBDIRECTORY(src/SupraLib)
IF(SUPRA_INTERFACE_COMMANDLINE)
ADD_SUBDIRECTORY(src/CommandlineInterface)
ENDIF(SUPRA_INTERFACE_COMMANDLINE)
IF(SUPRA_INTERFACE_EXECUTOR)
ADD_SUBDIRECTORY(src/SupraExecutor)
ENDIF(SUPRA_INTERFACE_EXECUTOR)
IF(SUPRA_INTERFACE_GRAPHIC)
ADD_SUBDIRECTORY(src/GraphicInterface)
ENDIF(SUPRA_INTERFACE_GRAPHIC)
IF(SUPRA_INTERFACE_ROS)
ADD_SUBDIRECTORY(src/RosInterface)
ENDIF(SUPRA_INTERFACE_ROS)
IF(SUPRA_INTERFACE_REST)
ADD_SUBDIRECTORY(src/RestInterface)
ENDIF(SUPRA_INTERFACE_REST)
ADD_SUBDIRECTORY(src/Wrapper)
ADD_SUBDIRECTORY(doc)

IF(SUPRA_DOWNLOAD_SAMPLEDATA)
	if(NOT EXISTS ${PROJECT_BINARY_DIR}/data/mockData_linearProbe.zip)
		message(STATUS "Downloading sample data (2D). This may take a while.")
		file(DOWNLOAD
			"https://f000.backblazeb2.com/file/supra-sample-data/mockData_linearProbe.zip"
			"${PROJECT_BINARY_DIR}/data/mockData_linearProbe.zip"
			SHOW_PROGRESS
		)
	endif()
	if(NOT EXISTS ${PROJECT_BINARY_DIR}/data/SUPRA_3D_sampleData.zip)
		message(STATUS "Downloading sample data (3D). This may take a while.")
		file(DOWNLOAD
			"https://f000.backblazeb2.com/file/supra-sample-data/SUPRA_3D_sampleData.zip"
			"${PROJECT_BINARY_DIR}/data/SUPRA_3D_sampleData.zip"
			SHOW_PROGRESS
		)
	endif()
	
	if(NOT EXISTS ${PROJECT_BINARY_DIR}/data/mockData_extracted)
		message(STATUS "Extracting sample data")
		execute_process(COMMAND cmake -E tar xf mockData_linearProbe.zip 
			WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/data/" )
		execute_process(COMMAND cmake -E tar xf SUPRA_3D_sampleData.zip 
			WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/data/" )
		execute_process(COMMAND cmake -E touch mockData_extracted
			WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/data/" )
	ENDIF()
	file(COPY config/configDemo.xml DESTINATION "${PROJECT_BINARY_DIR}/data/")
	file(COPY config/configDemo3D.xml DESTINATION "${PROJECT_BINARY_DIR}/data/")

	INSTALL(FILES ${PROJECT_BINARY_DIR}/data/mockData_linearProbe.zip
      DESTINATION data
      COMPONENT applicationData)
    INSTALL(FILES ${PROJECT_BINARY_DIR}/data/SUPRA_3D_sampleData.zip
      DESTINATION data
      COMPONENT applicationData)

ENDIF(SUPRA_DOWNLOAD_SAMPLEDATA)

IF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")

	SET(CPACK_PACKAGING_INSTALL_PREFIX "/tmp")
	SET(CPACK_INSTALL_PREFIX "/tmp")
	SET(CPACK_TOPLEVEL_TAG "/tmp")
	SET(CPACK_PACKAGE_DEFAULT_LOCATION "/opt")
	SET(CPACK_OUTPUT_FILE_PREFIX binpackages)
	SET(CPACK_GENERATOR "DEB")
	SET(CPACK_SOURCE_GENERATOR "DEB")
	SET(CPACK_DEBIAN_PACKAGE_LICENSE "LGPL 2.1")


	SET(CPACK_PACKAGE_NAME "supra")
	SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open Source Software Defined Ultrasound Processing for Real-Time Applications")
	SET(CPACK_PACKAGE_VENDOR "Chair for Computer Aided Medical Procedures - TUM")
	SET(CPACK_PACKAGE_CONTACT "Ruediger Goebl")
	SET(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/IFL-CAMP/supra")
	SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
	SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
	SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
	SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

	SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libtbb-dev , qt5-default , libopenigtlink-dev , libcpprest-dev , libboost-all-dev")
	 
	SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
	SET(CPACK_DEBIAN_PACKAGE_SECTION "science")
	SET(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})

	SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/tools/postinst;${CMAKE_CURRENT_SOURCE_DIR}/copyright")

	SET(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "off")
	SET(CPACK_SOURCE_INSTALLED_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR};/opt/${CPACK_PACKAGE_NAME}")
	SET(CPACK_SOURCE_IGNORE_FILES "${CMAKE_SOURCE_DIR}/build/;${CMAKE_SOURCE_DIR}/.git/")

	SET(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
	SET(CPACK_COMPONENTS_ALL applications applicationData)
	# Do this last
	INCLUDE(CPack)

ENDIF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
