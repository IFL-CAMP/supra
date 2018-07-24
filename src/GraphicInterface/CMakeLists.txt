cmake_minimum_required( VERSION 3.0.0 FATAL_ERROR )
MESSAGE(STATUS "Building SUPRA Graphic Interface")

# if ITK is used we need to include this for each executable
include(supraIncludeITK)

# Widgets finds its own dependencies.
find_package(Qt5 REQUIRED Widgets)

include(ExternalProject)

SET(NodeEditor_DIR "${CMAKE_CURRENT_BINARY_DIR}/NodeEditor")
SET(NodeEditor_GIT_REPOSITORY "https://github.com/goeblr/nodeeditor.git" CACHE STRING "")
SET(NodeEditor_GIT_TAG "feature_lightweight" CACHE STRING "")
ExternalProject_Add( 
	NodeEditor
	PREFIX "${NodeEditor_DIR}"
	
	LOG_DOWNLOAD TRUE
	LOG_UPDATE TRUE
	LOG_CONFIGURE TRUE
	LOG_BUILD TRUE
	LOG_INSTALL TRUE
	
	SOURCE_DIR "${NodeEditor_DIR}"
	BINARY_DIR "${NodeEditor_DIR}_build"
	STAMP_DIR "${NodeEditor_DIR}_stamp"
	TMP_DIR "${NodeEditor_DIR}_tmp"
	#--Download step--------------
	GIT_REPOSITORY ${NodeEditor_GIT_REPOSITORY}
	GIT_TAG ${NodeEditor_GIT_TAG}
	#--Configure step-------------
	CMAKE_ARGS
	  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
	  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
	  -DCMAKE_BUILD_TYPE=Release
	  -DBUILD_EXAMPLES=OFF
	  -DQt5_DIR=${Qt5_DIR}
	  -DCMAKE_INSTALL_PREFIX=${NodeEditor_DIR}_install
	  -DNODE_EDITOR_STATIC=
	#--Build step-----------------
	#BUILD_ALWAYS 0
	#--Install step-----------------
	INSTALL_DIR=${NodeEditor_DIR}_install
)
SET(NodeEditor_LIBRARIES "nodes")
INCLUDE_DIRECTORIES(SUPRA_GUI "${NodeEditor_DIR}_install/include")
LINK_DIRECTORIES(SUPRA_GUI "${NodeEditor_DIR}_install/lib")


OPTION(SUPRA_USE_CAMPVIS "Use CAMPVis for data previews" OFF)
if(${SUPRA_USE_CAMPVIS})
	# The CAMPVis package should be found automatically if you built CAMPVis through CMake
	FIND_PACKAGE(CAMPVis REQUIRED)
	SET(CAMPVIS_SOURCES 
		CampvisPreviewReciever.cpp)
	SET(CAMPVIS_HEADERS
		CampvisPreviewReciever.h)
ELSE()
	SET(CAMPVIS_SOURCES)
	SET(CAMPVIS_HEADERS)
ENDIF(${SUPRA_USE_CAMPVIS})

set ( SOURCES
 main.cpp
 mainwindow.cpp
 parameterWidget.cpp
 parametersWidget.cpp
 previewWidget.cpp
 previewBuilderQt.cpp
 QImagePreviewReciever.cpp
 QTrackerPreviewReciever.cpp
 NodeExplorerDataModel.cpp
 ${CAMPVIS_SOURCES}
)

set ( HEADERS
 parameterWidget.h
 previewBuilderQt.h
 QImagePreviewReciever.h
 QTrackerPreviewReciever.h
 NodeExplorerDataModel.h
 ${CAMPVIS_HEADERS}
)

set ( MOC_HEADERS
 mainwindow.h
 parametersWidget.h
 previewWidget.h
)

set ( UIS
 mainwindow.ui
 parametersWidget.ui
 previewWidget.ui
)

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

QT5_WRAP_CPP(HEADER_MOC ${MOC_HEADERS})
QT5_WRAP_UI(FORMS ${UIS})

IF(${SUPRA_USE_CAMPVIS})
	# The available CAMPVis linking targets are campvis-core, campvis-modules, or campvis-application-lib
	SET(CAMPVIS_LIB campvis-modules)
	SET(CAMPVIS_DEFINITIONS "${CAMPVIS_DEFINITIONS};-DHAVE_CAMPVIS")
ELSE(${SUPRA_USE_CAMPVIS})
	SET(CAMPVIS_LIB)
	SET(CAMPVIS_INCLUDE_DIRS)
	SET(CAMPVIS_DEFINITIONS)
ENDIF(${SUPRA_USE_CAMPVIS})

INCLUDE_DIRECTORIES(SUPRA_GUI
	${SUPRA_Lib_INCLUDEDIRS}
	${CAMPVIS_INCLUDE_DIRS}
)
LINK_DIRECTORIES(SUPRA_GUI
	${SUPRA_Lib_LIBDIRS}
)

add_executable(SUPRA_GUI ${SOURCES} ${HEADERS} ${MOC_HEADERS} ${FORMS} ${UIS})

TARGET_COMPILE_DEFINITIONS(SUPRA_GUI
	PRIVATE ${SUPRA_Lib_DEFINES}
	${CAMPVIS_DEFINITIONS} NODE_EDITOR_STATIC)

if(NOT SUPRA_INTERFACE_GRAPHIC_CMD)
	# generate proper GUI program on specified platform
	if(WIN32) # Check if we are on Windows
		if(MSVC) # Check if we are using the Visual Studio compiler
			set_target_properties(SUPRA_GUI PROPERTIES
				WIN32_EXECUTABLE YES
				LINK_FLAGS "/ENTRY:mainCRTStartup"
			)
		elseif(CMAKE_COMPILER_IS_GNUCXX)
				# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mwindows") # Not tested
		else()
			message(SEND_ERROR "You are using an unsupported Windows compiler! (Not MSVC or GCC)")
		endif(MSVC)
	elseif(APPLE)
		set_target_properties(SUPRA_GUI PROPERTIES
				MACOSX_BUNDLE YES
		)
	elseif(UNIX)
		# Nothing special required
	else()
		message(SEND_ERROR "You are on an unsupported platform! (Not Win32, Mac OS X or Unix)")
	endif(WIN32)
endif(NOT SUPRA_INTERFACE_GRAPHIC_CMD)

TARGET_LINK_LIBRARIES(SUPRA_GUI
	${SUPRA_Lib_LIBRARIES}
	${CAMPVIS_LIB}
	${NodeEditor_LIBRARIES}
)
set_property(TARGET SUPRA_GUI PROPERTY CXX_STANDARD 11)
set_property(TARGET SUPRA_GUI PROPERTY CXX_STANDARD_REQUIRED ON)

add_dependencies(SUPRA_GUI SUPRA_Lib NodeEditor)

qt5_use_modules(SUPRA_GUI Widgets)
