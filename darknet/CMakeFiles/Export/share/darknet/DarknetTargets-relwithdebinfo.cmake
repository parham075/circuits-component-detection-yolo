#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Darknet::dark" for configuration "RelWithDebInfo"
set_property(TARGET Darknet::dark APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(Darknet::dark PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "C:/Users/ASUS/Desktop/yolov4/darknet/darknet.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "C:/Users/ASUS/Desktop/yolov4/darknet/darknet.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS Darknet::dark )
list(APPEND _IMPORT_CHECK_FILES_FOR_Darknet::dark "C:/Users/ASUS/Desktop/yolov4/darknet/darknet.lib" "C:/Users/ASUS/Desktop/yolov4/darknet/darknet.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
