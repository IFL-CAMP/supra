#----------------------------------------------------------------
# Generated CMake target import file for configuration "None".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cpprestsdk::cpprest" for configuration "None"
set_property(TARGET cpprestsdk::cpprest APPEND PROPERTY IMPORTED_CONFIGURATIONS NONE)
set_target_properties(cpprestsdk::cpprest PROPERTIES
  IMPORTED_LOCATION_NONE "${_IMPORT_PREFIX}/lib/aarch64-linux-gnu/libcpprest.so.2.10"
  IMPORTED_SONAME_NONE "libcpprest.so.2.10"
  )

list(APPEND _IMPORT_CHECK_TARGETS cpprestsdk::cpprest )
list(APPEND _IMPORT_CHECK_FILES_FOR_cpprestsdk::cpprest "${_IMPORT_PREFIX}/lib/aarch64-linux-gnu/libcpprest.so.2.10" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
