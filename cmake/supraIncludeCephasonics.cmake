# defines varaibles to build with the Cephasonics interface
#CEPHASONICS_INCLUDE
#CEPHASONICS_DEFINES
#CEPHASONICS_LIBRARIES
#CEPHASONICS_LIBDIRS

####################################################
#     System Platform Detection
####################################################
IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(MACH_TYPE x86_64)
ELSE()
	SET(MACH_TYPE x86_32)
ENDIF()

####################################################
#           Application Include Directories
####################################################

#SET(PROJ_HOME $ENV{PROJ_HOME})
SET(PROJ_HOME /usr/local/cusdk)
#MESSAGE(STATUS "Found Cephasonics software: $ENV{PROJ_HOME}")

SET(APP_INC_CORE
	${PROJ_HOME}/inc/fal/cmn
	${PROJ_HOME}/inc/fal/core
	${PROJ_HOME}/inc/fal/capdmp
	${PROJ_HOME}/inc/fal/scncnv
	${PROJ_HOME}/inc/fal/scncnv/cmd
	${PROJ_HOME}/inc/fal/xscan
	${PROJ_HOME}/inc/fal/advmde
	${PROJ_HOME}/inc/fal/advmde/cmd
	${PROJ_HOME}/inc/fal/core/cmd
	${PROJ_HOME}/inc/fal/imgenh/cmd
	${PROJ_HOME}/inc/fal/exschd/cmd)
	
#TODO this can probably be removed
#SET(APP_INC_UI ${PROJ_HOME}/inc/usr/ui)
            
#CEPHASONICS_DEFINES  = -fPIC
SET(CEPHASONICS_DEFINES _REENTRANT _NO_ACINT_)

####################################################
#           Application LIB Directories
####################################################             

SET(APP_LIBDIR ${PROJ_HOME}/lib/${MACH_TYPE}/cpp)

####################################################
#           Application Libraries
#################################################### 

SET(APP_CORE_LIBS us_fal us_ral us_util bf SCPP usb-1.0 xml_parser)
SET(APP_DEP_LIBS png pthread xml2 fftw3 m tomcrypt)
SET(APP_UI_LIBS us_ui)

####################################################
#           Cqlink Includes Path
#################################################### 

#CQLINK_INC  =-I ${PROJ_HOME}/inc/cqlink/client \
#             -I ${PROJ_HOME}/inc/cqlink/network \
#             -I ${PROJ_HOME}/inc/cqlink/msg
             
####################################################
#           Cqlink Libraries
####################################################

#CQLINK_SRV_LIBS = -lus_cqlink_srv -lus_cqlink_msg -lus_cqlink_nw -lus_fal -lSCPP -lbf -lus_ral -lus_util -lusb-1.0 -lpthread -lm -lprotobuf -lfftw3 -lxml_parser -lxml2 -ltomcrypt

#CQLINK_CLI_LIBS  = -lus_cqlink_cli -lus_cqlink_msg -lus_cqlink_nw -lus_util -lprotobuf

####################################################
#           Cqlink LIB Directories
#################################################### 
#CQLINK_LIBDIR = $(APP_LIBDIR) 

#IF(MAC)
#CEPHASONICS_INCLUDE += -I /opt/local/include
#CEPHASONICS_INCLUDE += -I /opt/local/include/boost
#CEPHASONICS_INCLUDE += -I /opt/local/include/boost/thread
#CEPHASONICS_LIBDIRS += -L /opt/local/lib


SET(CEPHASONICS_INCLUDE /usr/include/)

set(Boost_USE_STATIC_LIBS      OFF)
set(Boost_USE_MULTITHREADED    ON)
set(Boost_USE_STATIC_RUNTIME   OFF)

# determine whether we are building on ubuntu and which version
find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -rs
    OUTPUT_VARIABLE LSB_RELEASE_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(LSB_RELEASE_ID_SHORT EQUAL "Ubuntu" AND LSB_RELEASE_SHORT EQUAL "16.04")
  find_package(Boost 1.54.0 EXACT REQUIRED COMPONENTS system filesystem program_options thread)
ELSE()
  find_package(Boost REQUIRED COMPONENTS system filesystem program_options thread)
ENDIF()
if(Boost_FOUND)
  SET(CEPHASONICS_INCLUDE ${CEPHASONICS_INCLUDE} ${Boost_INCLUDE_DIR} ${Boost_INCLUDE_DIR}/boost ${Boost_INCLUDE_DIR}/boost/thread)
  SET(CEPHASONICS_LIBRARIES ${CEPHASONICS_LIBRARIES} ${Boost_LIBRARIES})
  SET(CEPHASONICS_LIBDIRS ${CEPHASONICS_LIBDIRS} ${Boost_LIBRARY_DIRS})
endif()
	
SET(CEPHASONICS_INCLUDE ${CEPHASONICS_INCLUDE} ${APP_INC_CORE})# ${APP_INC_UI})
SET(CEPHASONICS_LIBRARIES ${CEPHASONICS_LIBRARIES} ${APP_CORE_LIBS} ${APP_UI_LIBS})
SET(CEPHASONICS_LIBDIRS ${CEPHASONICS_LIBDIRS} ${APP_LIBDIR})



