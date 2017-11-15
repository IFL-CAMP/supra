CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA ROS Interface")

# if ITK is used we need to include this for each executable
include(supraIncludeITK)

# Add rosserial for this project
include(supraIncludeRos)

############################################
#Program base source files
SET(SUPRA_ROS_SOURCE 
	main.cpp
	RosInterface.cpp
	${ROSSERIAL_SOURCES})
SET(SUPRA_ROS_HEADERS
	RosInterface.h
	${ROSSERIAL_HEADERS})
	
############################################
#Build Commandline interface
SOURCE_GROUP(src FILES ${SUPRA_ROS_SOURCE})
SOURCE_GROUP(inc FILES ${SUPRA_ROS_HEADERS})

INCLUDE_DIRECTORIES(SUPRA_ROS
	${SUPRA_Lib_INCLUDEDIRS}
	${ROSSERIAL_INCLUDE}
)
LINK_DIRECTORIES(SUPRA_ROS
	${SUPRA_Lib_LIBDIRS}
)

ADD_EXECUTABLE(SUPRA_ROS
	${SUPRA_ROS_SOURCE}
	${SUPRA_ROS_HEADERS}
)
TARGET_LINK_LIBRARIES(SUPRA_ROS
	${SUPRA_Lib_LIBRARIES}
)
TARGET_COMPILE_DEFINITIONS(SUPRA_ROS
	PRIVATE ${SUPRA_Lib_DEFINES})
set_property(TARGET SUPRA_ROS PROPERTY CXX_STANDARD 11)
set_property(TARGET SUPRA_ROS PROPERTY CXX_STANDARD_REQUIRED ON)

add_dependencies(SUPRA_ROS SUPRA_Lib)
