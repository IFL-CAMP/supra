CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA Commandline Interface")

# if ITK is used we need to include this for each executable
include(supraIncludeITK)

############################################
#Program base source files
SET(SUPRA_CMD_SOURCE 
	main.cpp
	CommandlineInterface.cpp)
SET(SUPRA_CMD_HEADERS
	CommandlineInterface.h)
	
############################################
#Build Commandline interface
SOURCE_GROUP(src FILES ${SUPRA_CMD_SOURCE})
SOURCE_GROUP(inc FILES ${SUPRA_CMD_HEADERS})

INCLUDE_DIRECTORIES(SUPRA_CMD
	${SUPRA_Lib_INCLUDEDIRS}
)
LINK_DIRECTORIES(SUPRA_CMD
	${SUPRA_Lib_LIBDIRS}
)

ADD_EXECUTABLE(SUPRA_CMD
	${SUPRA_CMD_SOURCE}
	${SUPRA_CMD_HEADERS}
)
TARGET_COMPILE_DEFINITIONS(SUPRA_CMD
	PRIVATE ${SUPRA_Lib_DEFINES})
TARGET_LINK_LIBRARIES(SUPRA_CMD
	${SUPRA_Lib_LIBRARIES}
)
set_property(TARGET SUPRA_CMD PROPERTY CXX_STANDARD 11)
set_property(TARGET SUPRA_CMD PROPERTY CXX_STANDARD_REQUIRED ON)

add_dependencies(SUPRA_CMD SUPRA_Lib)
