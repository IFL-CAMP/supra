IF(WIN32)
    SET(ROSSERIAL_DIR "${CMAKE_SOURCE_DIR}/rosserial_libs/ros_lib")
    SET(ROS_HEADERS
        "${ROSSERIAL_DIR}/WindowsSocket.h"
        "${ROSSERIAL_DIR}/ros.h")

    IF(${ROSSERIAL_HAVE_SOURCE_FILES})
        SET(ROS_SOURCES)
    ELSE()
        SET(ROS_SOURCES
            "${ROSSERIAL_DIR}/time.cpp"
            "${ROSSERIAL_DIR}/WindowsSocket.cpp"
            "${ROSSERIAL_DIR}/duration.cpp")
    ENDIF()

    SET(ROS_INCLUDE ${ROSSERIAL_DIR})
    SET(ROS_LIBRARIES)
    SET(ROS_DEFINES ROS_ROSSERIAL)
ELSE()
    find_package(roscpp REQUIRED)
    find_package(geometry_msgs REQUIRED)

    SET(ROS_HEADERS)
    SET(ROS_SOURCES)
    SET(ROS_INCLUDE ${roscpp_INCLUDE_DIRS} ${geometry_msgs_INCLUDE_DIRS})
    IF(SUPRA_DEVICE_ROS_IMAGE_OUT OR SUPRA_INTERFACE_ROS)
        find_package(supra_msgs REQUIRED)
        SET(ROS_INCLUDE ${ROS_INCLUDE} ${supra_msgs_INCLUDE_DIRS})
    ENDIF()
    SET(ROS_LIBRARIES ${roscpp_LIBRARIES})
    SET(ROS_DEFINES ROS_REAL)
ENDIF()

SET(ROS_HEADERS ${ROS_HEADERS}
    utilities/RosWrapper.h)
IF(NOT (${ROSSERIAL_HAVE_SOURCE_FILES}))
    SET(ROS_SOURCES ${ROS_SOURCES}
        utilities/RosWrapper.cpp)
    #remember that we already added the source files
    SET(ROSSERIAL_HAVE_SOURCE_FILES TRUE)
ENDIF()

SET(ROS_DEFINES ${ROS_DEFINES} ROS_PRESENT)
SET(ROS_FOUND TRUE)
