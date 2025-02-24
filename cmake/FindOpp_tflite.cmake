# opp_tflite/cmake/FindOpp_tflite.cmake
#  here are lib paths
# /home/nishi/usr/local/lib/tensorflow-lite /home/nishi/usr/local/lib/tensorflow-lite-flex
# https://qiita.com/shohirose/items/d9bda00a39a113965c5c
# /usr/share/cmake-3.22/Modules/FindLAPACK.cmake

# ROS2 だと、 opp_tflite.hpp が、tensorflow-lite を参照するので、そのパスも必要みたい。
set(Opp_tflite_INCLUDE_DIR "/home/nishi/usr/local/include/opp_tflite" "/home/nishi/usr/local/include/tensorflow-lite")
set(Opp_tflite_LIBRARY 
  "-L/home/nishi/usr/local/lib/opp_tflite -lopp_tflite"
)
#message(STATUS "Opp_tflite_LIBRARY=${Opp_tflite_LIBRARY}")

mark_as_advanced(
  Opp_tflite_INCLUDE_DIR
  Opp_tflite_LIBRARY     # ヘッダーのみのライブラリの場合は不要
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Opp_tflite
  REQUIRED_VARS
    Opp_tflite_INCLUDE_DIR
    Opp_tflite_LIBRARY      # ヘッダーのみのライブラリの場合は不要
  )

if(Opp_tflite_FOUND AND NOT TARGET Opp_tflite::Opp_tflite)
  add_library(Opp_tflite::Opp_tflite UNKNOWN IMPORTED)
  set_target_properties(Opp_tflite::Opp_tflite PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"  # ヘッダーのみのライブラリの場合は不要
    IMPORTED_LOCATION "${Opp_tflite_LIBRARY}"      # ヘッダーのみのライブラリの場合は不要
    INTERFACE_INCLUDE_DIRECTORIES "${Opp_tflite_INCLUDE_DIR}"
    )
endif()
