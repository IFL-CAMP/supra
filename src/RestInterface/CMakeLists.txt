cmake_minimum_required( VERSION 3.0.0 FATAL_ERROR )

MESSAGE(STATUS "Building SUPRA REST Interface")

MESSAGE(STATUS "Uses the cpp REST api from https://github.com/Microsoft/cpprestsdk")

# This assumes installation of cpprestsdk through apt-get
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  find_file (UBUNTU_ISSUE issue PATHS /etc)
  if (UBUNTU_ISSUE)
    file (STRINGS ${UBUNTU_ISSUE} UBUNTU_XENIAL REGEX "Ubuntu 16.04")
    if (UBUNTU_XENIAL)
      SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    endif()
  endif()
endif()
find_package(cpprestsdk REQUIRED)

# Configure required libraries ...
if(UNIX) # Darwing or Linux
    
    find_package(Boost REQUIRED COMPONENTS system thread log program_options chrono)
    find_package(Threads REQUIRED)

    if(APPLE)
        # Prefer a homebrew version of OpenSSL over the one in /usr/lib
	    file(GLOB OPENSSL_ROOT_DIR /usr/local/Cellar/openssl/*)
        # Prefer the latest (make the latest one first)
	    list(REVERSE OPENSSL_ROOT_DIR)

        find_package(OpenSSL 1.0.2 REQUIRED)
        set(OPENSSL_VERSION "1.0.2p")
    else()
        find_package(OpenSSL 1.0.1 REQUIRED)
        set(OPENSSL_VERSION "1.0.1")
    endif()

elseif(WIN32) # Windows systems including Win64.
    message(FATAL_ERROR "-- Windows is not supported for now.")
else()
    message(FATAL_ERROR "-- Unsupported platform sorry! :( ")
endif()

# Configure compiler options ...
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    message("-- configuring clang options")
    set(CMAKE_CXX_FLAGS "-arch x86_64 -std=c++11 -stdlib=libc++ -DBOOST_LOG_DYN_LINK -Wno-deprecated-declarations")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")

    message("-- configuring gcc options")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -DBOOST_LOG_DYN_LINK")

endif()

# headers search paths ...
set(SUPRA_REST_INCLUDEDIRS ${CPPRESTSDK_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${OPENSSL_INCLUDE_DIR})

# library search paths ...
if(APPLE)
    set(OPENSSL_LIBS "/usr/local/Cellar/openssl/${OPENSSL_VERSION}/lib/libssl.1.0.0.dylib;/usr/local/Cellar/openssl/${OPENSSL_VERSION}/lib/libcrypto.1.0.0.dylib")
    set(CPPRESTSDK_LIBRARY "${PROJECT_SOURCE_DIR}/libs/cpprestsdk/build.release/Binaries/libcpprest.a")
    set(ZIP_LIBRARY "/usr/local/Cellar/zlib/1.2.11/lib/libz.dylib")

    set(LIBRARIES_SEARCH_PATHS ${OPENSSL_LIBS} ${Boost_LIBRARIES} ${CPPRESTSDK_LIBRARY} ${ZIP_LIBRARY})
else()
    set(OPENSSL_LIBS "${OPENSSL_LIBRARIES}")
    set(CPPRESTSDK_LIBRARY cpprestsdk::cpprest)

    set(LIBRARIES_SEARCH_PATHS ${CPPRESTSDK_LIBRARY} ${OPENSSL_LIBS} ${Boost_LIBRARIES})
endif()

message(BOOST_LIBS " ${Boost_LIBRARIES}")
message(OPENSSL_LIBS " ${OPENSSL_LIBRARIES}")
message(CPPRESTSDK_LIBRARY " ${CPPRESTSDK_LIBRARY}")
message(LIBRARIES_SEARCH_PATHS " ${LIBRARIES_SEARCH_PATHS}")

#include_directories(${HEADER_SEARCH_PATHS})
#if (APPLE)
#    target_link_libraries(${PROJECT_NAME} "-framework CoreFoundation")
#    target_link_libraries(${PROJECT_NAME} "-framework Security")
#    target_link_libraries(${PROJECT_NAME} ${LIBRARIES_SEARCH_PATHS})
#    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-W1, -F/Library/Frameworks")
#else()
#    target_link_libraries(${PROJECT_NAME} ${LIBRARIES_SEARCH_PATHS})
#endif()



############################################
#Program base source files
SET(SUPRA_REST_SOURCE 
	main.cpp
	RestInterface.cpp
	foundation/network_utils.cpp
	foundation/basic_controller.cpp)
SET(SUPRA_REST_HEADERS
	RestInterface.h
	foundation/include/basic_controller.hpp
	foundation/include/controller.hpp
	foundation/include/network_utils.hpp
	foundation/include/runtime_utils.hpp
	foundation/include/std_micro_service.hpp
	foundation/include/usr_interrupt_handler.hpp)
	
############################################
#Build Commandline interface
SOURCE_GROUP(src FILES ${SUPRA_REST_SOURCE})
SOURCE_GROUP(inc FILES ${SUPRA_REST_HEADERS})

INCLUDE_DIRECTORIES(SUPRA_REST
	${SUPRA_Lib_INCLUDEDIRS}
	${SUPRA_REST_INCLUDEDIRS}
	"${CMAKE_CURRENT_SOURCE_DIR}/foundation/include"
)
LINK_DIRECTORIES(SUPRA_REST
	${SUPRA_Lib_LIBDIRS}
)

ADD_EXECUTABLE(SUPRA_REST
	${SUPRA_REST_SOURCE}
	${SUPRA_REST_HEADERS}
)
TARGET_LINK_LIBRARIES(SUPRA_REST
	${SUPRA_Lib_LIBRARIES}
	${LIBRARIES_SEARCH_PATHS}
)
TARGET_COMPILE_DEFINITIONS(SUPRA_REST
	PRIVATE ${SUPRA_Lib_DEFINES})
set_property(TARGET SUPRA_REST PROPERTY CXX_STANDARD 11)
set_property(TARGET SUPRA_REST PROPERTY CXX_STANDARD_REQUIRED ON)

add_dependencies(SUPRA_REST SUPRA_Lib)

INSTALL(TARGETS SUPRA_REST
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
