CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA Wrapper")

############################################
#Program base source files
SET(SUPRA_Wrapper_SOURCE 
	main.cpp)
SET(SUPRA_Wrapper_HEADERS)
	
############################################
#Build Wrappers
SOURCE_GROUP(src FILES ${SUPRA_Wrapper_SOURCE})
SOURCE_GROUP(inc FILES ${SUPRA_Wrapper_HEADERS})

INCLUDE_DIRECTORIES(SUPRA_Wrapper
	${SUPRA_Lib_INCLUDEDIRS}
)
LINK_DIRECTORIES(SUPRA_Wrapper
	${SUPRA_Lib_LIBDIRS}
)

ADD_EXECUTABLE(SUPRA_Wrapper
	${SUPRA_Wrapper_SOURCE}
	${SUPRA_Wrapper_HEADERS}
)
TARGET_COMPILE_DEFINITIONS(SUPRA_Wrapper
	PRIVATE ${SUPRA_Lib_DEFINES})
TARGET_LINK_LIBRARIES(SUPRA_Wrapper
	${SUPRA_Lib_LIBRARIES}
)
set_property(TARGET SUPRA_Wrapper PROPERTY CXX_STANDARD 11)
set_property(TARGET SUPRA_Wrapper PROPERTY CXX_STANDARD_REQUIRED ON)

add_dependencies(SUPRA_Wrapper SUPRA_Lib)
