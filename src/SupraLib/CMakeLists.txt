CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA_Lib")

SET(SUPRA_Lib_DEFINES)
IF(WIN32)
	# In some places we need to use insecure iterators. Let visual studio not warn about that
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES} _SCL_SECURE_NO_WARNINGS)
ENDIF(WIN32)

# find TBB (>= V4.0.0) always needs to be there
IF(WIN32)	
	SET(TBB_ROOT $ENV{TBBROOT})
	MESSAGE( STATUS "Looking for TBB in '${TBB_ROOT}' (environment variable 'TBBROOT')")
	SET(TBB_INCLUDE ${TBB_ROOT}/include)
	SET(TBB_LIBS ${TBB_ROOT}/lib)
	SET(TBB_LIBRARIES)
	IF(CMAKE_CL_64)
		SET(TBB_LIBS ${TBB_LIBS}/intel64/vc14)
	ELSE(CMAKE_CL_64)
		SET(TBB_LIBS ${TBB_LIBS}/ia32/vc14)
	ENDIF(CMAKE_CL_64)
ELSE()
	SET(TBB_ROOT ${SUPRA_EXTERNALS_TBB})
	MESSAGE( STATUS "Looking for TBB in '${TBB_ROOT}'")
	SET(TBB_INCLUDE ${TBB_ROOT}/include)
	SET(TBB_LIBS ${SUPRA_EXTERNALS_TBB_LIBS})
	SET(TBB_LIBRARIES tbb pthread)
ENDIF(WIN32)

# find Ultrasonix interface
IF (SUPRA_DEVICE_ULTRASONIX)
	IF(WIN32)
		IF(CMAKE_CL_64)
			MESSAGE(FATAL_ERROR "The Ultrasonix API can only be built on Win32")
		ENDIF(CMAKE_CL_64)
	ELSE(WIN32)
		MESSAGE(FATAL_ERROR "The Ultrasonix API can only be built on Win32")
	ENDIF(WIN32)
	SET( ULTRASONIX_SDK_DIR "${SUPRA_EXTERNALS}/Ultrasonix_${SUPRA_DEVICE_ULTRASONIX_V}")
	
	# external library ulterius
	add_library( ULTERIUS STATIC IMPORTED)

	IF (${SUPRA_DEVICE_ULTRASONIX_V} MATCHES "5.7")
		SET(UltrasonixHandler_LIBNAME ulterius)
	ELSEIF (${SUPRA_DEVICE_ULTRASONIX_V} MATCHES "6.07")
		SET(UltrasonixHandler_LIBNAME ulterius_old)
	ENDIF()
	SET(UltrasonixHandler_LIB_DIRS "${ULTRASONIX_SDK_DIR}/ulterius/lib/")
	SET(UltrasonixHandler_INCLUDE_DIRS "${ULTRASONIX_SDK_DIR}/ulterius/inc" )
ENDIF()

IF (SUPRA_DEVICE_CEPHASONICS)
	IF(WIN32)
		MESSAGE(FATAL_ERROR "The Cephasonics interface can only be built on linux")
	ENDIF(WIN32)

	INCLUDE(supraIncludeCephasonics)
ENDIF()

IF(SUPRA_CUDA)
	include(supraIncludeCuda)
	set(CUDA_SEPARABLE_COMPILATION ON)
	MESSAGE(STATUS "Found Cuda Version " ${CUDA_VERSION_STRING})
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;--use_fast_math;--default-stream=per-thread;-lineinfo")
	IF(WIN32)
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 5.0)
	ELSE()
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
	ENDIF()
    LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
	IF(NOT WIN32)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--std=c++11")
	ENDIF()
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_CUDA)
	SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS}
		${CUDA_INCLUDE_DIRS})
ENDIF(SUPRA_CUDA)

#find IGTL Interface
IF(SUPRA_DEVICE_IGTL_OUTPUT OR SUPRA_DEVICE_TRACKING_IGTL)
find_package(OpenIGTLink REQUIRED)
ENDIF(SUPRA_DEVICE_IGTL_OUTPUT OR SUPRA_DEVICE_TRACKING_IGTL)

IF(SUPRA_DEVICE_ITK_FILE_OUT)
	#SET(SUPRA_Lib_NEED_ITK TRUE)
	include(supraIncludeITK)
ENDIF(SUPRA_DEVICE_ITK_FILE_OUT)

IF(SUPRA_DEVICE_ROS_IMAGE_OUT OR SUPRA_DEVICE_TRACKING_ROS)
	# Add rosserial for this project
	include(supraIncludeRos)
ENDIF(SUPRA_DEVICE_ROS_IMAGE_OUT OR SUPRA_DEVICE_TRACKING_ROS)

IF(SUPRA_PROFILING)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		LOG_PROFILING)
ENDIF(SUPRA_PROFILING)

############################################
#lib base source files
SET(SUPRA_Lib_SOURCE
	ContainerFactory.cpp
	SupraManager.cpp
	RecordObject.cpp
	SyncRecordObject.cpp
	TrackerData.cpp
	TrackerDataSet.cpp
	USImageProperties.cpp
	ConfigurationDictionary.cpp
	ValueRangeDictionary.cpp
	StreamSynchronizer.cpp
	TemporalOffsetNode.cpp
	InterfaceFactory.cpp
	utilities/utility.cpp
	utilities/Logging.cpp
	utilities/CallFrequency.cpp
	utilities/TemplateTypeDefault.cpp
	utilities/SingleThreadTimer.cpp
	utilities/tinyxml2/tinyxml2.cpp
	FrequencyLimiterNode.cpp
	StreamSyncNode.cpp
	AutoQuitNode.cpp)
SET(SUPRA_Lib_HEADERS
	vec.h
	Container.h
	ContainerFactory.h
	SupraManager.h
	RecordObject.h
	SyncRecordObject.h
	TrackerData.h
	TrackerDataSet.h
	USImage.h
	USImageProperties.h
	ConfigurationDictionary.h
	ValueRangeDictionary.h
	AbstractNode.h
	AbstractInput.h
	AbstractOutput.h
	StreamSynchronizer.h
	TemporalOffsetNode.h
	InterfaceFactory.h
	utilities/utility.h
	utilities/cudaUtility.h
	utilities/Logging.h
	utilities/CallFrequency.h
	utilities/TemplateTypeDefault.h
	utilities/SingleThreadTimer.h
	utilities/FirFilterFactory.h
	utilities/tinyxml2/tinyxml2.h
	FrequencyLimiterNode.h
	StreamSyncNode.h
	AutoQuitNode.h)
	
SET(SUPRA_Lib_INCLUDEDIRS
	${SUPRA_Lib_INCLUDEDIRS}
	${TBB_INCLUDE}
	${CMAKE_CURRENT_SOURCE_DIR})
SET(SUPRA_Lib_LIBDIRS 
	${TBB_LIBS})
SET(SUPRA_Lib_LIBRARIES
	${TBB_LIBRARIES})
	
IF(ROS_FOUND)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		${ROS_SOURCES})
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		${ROS_HEADERS})
	SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS}
		${ROS_INCLUDE})
	SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES}
		${ROS_LIBRARIES})
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		${ROS_DEFINES})
ENDIF(ROS_FOUND)
	
############################################
#Lib modules
IF(SUPRA_BEAMFORMER)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		Beamformer/Sequencer.cpp
		Beamformer/Beamformer.cpp
		Beamformer/RxBeamformerParameters.cpp
		Beamformer/WindowFunction.cpp
		Beamformer/USTransducer.cpp
		Beamformer/BeamformingNode.cpp
		Beamformer/IQDemodulatorNode.cpp
		Beamformer/LogCompressorNode.cpp
		Beamformer/ScanConverterNode.cpp
		Beamformer/TemporalFilterNode.cpp
		Beamformer/RawDelayNode.cpp
		InputOutput/UltrasoundInterfaceRawDataMock.cpp)
	SET(SUPRA_Lib_CUDASOURCE ${SUPRA_Lib_CUDASOURCE}
		Beamformer/RxBeamformerCuda.cu
		Beamformer/IQDemodulator.cu
		Beamformer/LogCompressor.cu
		Beamformer/ScanConverter.cu
		Beamformer/TemporalFilter.cu
		Beamformer/RawDelay.cu)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		Beamformer/Sequencer.h
		Beamformer/Beamformer.h
		Beamformer/RxBeamformerCuda.h
		Beamformer/RxBeamformerCommon.h
		Beamformer/RxSampleBeamformerDelayAndSum.h
		Beamformer/RxSampleBeamformerDelayAndStdDev.h
		Beamformer/RxSampleBeamformerTestSignal.h
		Beamformer/RxBeamformerParameters.h
		Beamformer/WindowFunction.h
		Beamformer/USTransducer.h
		Beamformer/BeamformingNode.h
		Beamformer/IQDemodulator.h
		Beamformer/IQDemodulatorNode.h
		Beamformer/LogCompressor.h
		Beamformer/LogCompressorNode.h
		Beamformer/ScanConverter.h
		Beamformer/ScanConverterNode.h
		Beamformer/TemporalFilter.h
		Beamformer/TemporalFilterNode.h
		Beamformer/USRawData.h
		Beamformer/RawDelayNode.h
		Beamformer/RawDelay.h
		InputOutput/UltrasoundInterfaceRawDataMock.h)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_BEAMFORMER)
	SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES}
		${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
	
	IF(DEFINED CUDA_CUBLAS_LIBRARIES)
		SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
			Beamformer/BeamformingMVNode.h)
		SET(SUPRA_Lib_CUDASOURCE ${SUPRA_Lib_CUDASOURCE}
			Beamformer/BeamformingMVNode.cu)
		SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
			HAVE_BEAMFORMER_MINIMUM_VARIANCE)
	ELSE()
		MESSAGE(STATUS "CUBLAS found, NOT adding minimum variance beamformer")
	ENDIF()
ENDIF(SUPRA_BEAMFORMER)
IF(SUPRA_DEVICE_ULTRASOUND_SIM)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/UltrasoundInterfaceSimulated.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/UltrasoundInterfaceSimulated.h)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_ULTRASOUND_SIM)
ENDIF()
IF(SUPRA_DEVICE_TRACKING_SIM)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/TrackerInterfaceSimulated.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/TrackerInterfaceSimulated.h)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_TRACKING_SIM)
ENDIF()
IF(SUPRA_DEVICE_TRACKING_IGTL)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/TrackerInterfaceIGTL.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/TrackerInterfaceIGTL.h)
	SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS}
		${OpenIGTLink_INCLUDE_DIRS})
	SET(SUPRA_Lib_LIBDIRS ${SUPRA_Lib_LIBDIRS}
		${OpenIGTLink_LIBRARY_DIRS})
	SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES}
		${OpenIGTLink_LIBRARIES})
	IF(WIN32)
		SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES} ws2_32 wsock32)
	ENDIF(WIN32)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_TRACKING_IGTL)
ENDIF()
IF(SUPRA_DEVICE_TRACKING_ROS)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/TrackerInterfaceROS.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/TrackerInterfaceROS.h)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_TRACKING_ROS)
ENDIF()
IF(SUPRA_DEVICE_ULTRASONIX)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/UltrasoundInterfaceUltrasonix.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/UltrasoundInterfaceUltrasonix.h)
	SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS}
		${UltrasonixHandler_INCLUDE_DIRS})
	SET(SUPRA_Lib_LIBDIRS ${SUPRA_Lib_LIBDIRS}
		${UltrasonixHandler_LIB_DIRS})
	SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES}
		${UltrasonixHandler_LIBNAME})
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_ULTRASONIX)
ENDIF()
IF(SUPRA_DEVICE_IGTL_OUTPUT)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/OpenIGTLinkOutputDevice.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/OpenIGTLinkOutputDevice.h)
	SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS}
		${OpenIGTLink_INCLUDE_DIRS})
	SET(SUPRA_Lib_LIBDIRS ${SUPRA_Lib_LIBDIRS}
		${OpenIGTLink_LIBRARY_DIRS})
	SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES}
		${OpenIGTLink_LIBRARIES})
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_IGTL_OUTPUT)
ENDIF()
IF(SUPRA_DEVICE_CEPHASONICS)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/UsIntCephasonicsBmode.cpp
		InputOutput/UsIntCephasonicsBmodeProc.cpp
		InputOutput/UsIntCephasonicsBtcc.cpp
		InputOutput/UsIntCephasonicsBtccProc.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/UsIntCephasonicsBmode.h
		InputOutput/UsIntCephasonicsBmodeProc.h
		InputOutput/UsIntCephasonicsBtcc.h
		InputOutput/UsIntCephasonicsBtccProc.h)

	IF(SUPRA_CUDA)
		SET(SUPRA_Lib_CUDASOURCE ${SUPRA_Lib_CUDASOURCE}
			InputOutput/UsIntCephasonicsCc.cu
			InputOutput/UsIntCephasonicsCcProc.cpp)
		SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
			InputOutput/UsIntCephasonicsCc.h
			InputOutput/UsIntCephasonicsCcProc.h)
	ENDIF(SUPRA_CUDA)

	SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS}
		${CEPHASONICS_INCLUDE})
	SET(SUPRA_Lib_LIBDIRS ${SUPRA_Lib_LIBDIRS}
		${CEPHASONICS_LIBDIRS})
	SET(SUPRA_Lib_LIBRARIES ${SUPRA_Lib_LIBRARIES}
		${CEPHASONICS_LIBRARIES})
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_CEPHASONICS
		${CEPHASONICS_DEFINES}) #needed for cephasonics interface
ENDIF()
IF(SUPRA_DEVICE_ITK_FILE_OUT)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/MetaImageOutputDevice.cpp
		InputOutput/MhdSequenceWriter.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/MetaImageOutputDevice.h
		InputOutput/MhdSequenceWriter.h)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_METAIMAGE_OUTPUT)
ENDIF()
IF(SUPRA_DEVICE_ROS_IMAGE_OUT)
	SET(SUPRA_Lib_SOURCE ${SUPRA_Lib_SOURCE}
		InputOutput/RosImageOutputDevice.cpp)
	SET(SUPRA_Lib_HEADERS ${SUPRA_Lib_HEADERS}
		InputOutput/RosImageOutputDevice.h)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		HAVE_DEVICE_ROSIMAGE_OUTPUT)
ENDIF()

IF(WIN32)
	SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES}
		_CRT_SECURE_NO_WARNINGS)
ENDIF(WIN32)
############################################
#Build lib
SOURCE_GROUP(src FILES ${SUPRA_Lib_SOURCE} ${SUPRA_Lib_CUDASOURCE})
SOURCE_GROUP(inc FILES ${SUPRA_Lib_HEADERS})

INCLUDE_DIRECTORIES(SUPRA_Lib
	${SUPRA_Lib_INCLUDEDIRS})

IF(CUDA_FOUND)
CUDA_ADD_LIBRARY(SUPRA_Lib
	${SUPRA_Lib_SOURCE}
	${SUPRA_Lib_HEADERS}
	${SUPRA_Lib_CUDASOURCE}
)
ELSE ()
ADD_LIBRARY(SUPRA_Lib
	${SUPRA_Lib_SOURCE}
	${SUPRA_Lib_HEADERS}
	${SUPRA_Lib_CUDASOURCE}
)
ENDIF(CUDA_FOUND)
TARGET_INCLUDE_DIRECTORIES(SUPRA_Lib
	PUBLIC ${SUPRA_Lib_INCLUDEDIRS}
)
LINK_DIRECTORIES(SUPRA_Lib
	${SUPRA_Lib_LIBDIRS}
)

TARGET_COMPILE_DEFINITIONS(SUPRA_Lib 
	PRIVATE ${SUPRA_Lib_DEFINES})
set_property(TARGET SUPRA_Lib PROPERTY CXX_STANDARD 11)
set_property(TARGET SUPRA_Lib PROPERTY CXX_STANDARD_REQUIRED ON)

SET(SUPRA_Lib_INCLUDEDIRS ${SUPRA_Lib_INCLUDEDIRS} PARENT_SCOPE)
SET(SUPRA_Lib_LIBDIRS ${SUPRA_Lib_LIBDIRS} PARENT_SCOPE)
SET(SUPRA_Lib_LIBRARIES $<TARGET_FILE:SUPRA_Lib> ${SUPRA_Lib_LIBRARIES} PARENT_SCOPE)
SET(SUPRA_Lib_DEFINES ${SUPRA_Lib_DEFINES} PARENT_SCOPE)

IF(${SUPRA_Lib_NEED_ITK})
	SET(SUPRA_Lib_NEED_ITK TRUE PARENT_SCOPE)
ENDIF()
IF(${ROSSERIAL_HAVE_SOURCE_FILES})
	SET(ROSSERIAL_HAVE_SOURCE_FILES TRUE PARENT_SCOPE)
ENDIF()
