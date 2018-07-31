#get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
#SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}")
find_package(CUDA REQUIRED)
MESSAGE(STATUS "Using workaround for FindCUDA.cmake quote handling. In current version (3.9.0) it is still present. See https://gitlab.kitware.com/cmake/cmake/issues/16510 for details.")
