cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_jetson)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_recognition_msgs
  image_recognition_util
  image_transport
  roscpp
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_recognition_msgs image_recognition_util image_transport roscpp sensor_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

find_package(jetson-utils)
find_package(jetson-inference)
if(jetson-inference_FOUND) # If we are on Jetson, use the actual impl, otherwise a mock
    find_package(CUDA REQUIRED)
    find_package(Qt4 REQUIRED)
    include(${QT_USE_FILE})
    add_definitions(${QT_DEFINITIONS})
    include_directories(${CUDA_INCLUDE_DIRS})
    add_library(detect_net src/detect_net.cpp)
    target_link_libraries(detect_net ${catkin_LIBRARIES} jetson-inference)
else()
  message(WARNING "Jetson-inference not found, using a mock instead ...")
  message(WARNING "The library can be installed using scripts/install_jetson_inference.bash")
  add_library(detect_net src/detect_net_mock.cpp)
  target_link_libraries(detect_net ${catkin_LIBRARIES})
endif()
add_dependencies(detect_net ${catkin_EXPORTED_TARGETS})

add_executable(detect_net_ros src/detect_net_ros.cpp)
target_link_libraries(detect_net_ros detect_net)

install(TARGETS
  detect_net_ros
  detect_net
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/install_jetson_inference.bash
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if (CATKIN_ENABLE_TESTING)
    # Test catkin lint
    find_program(CATKIN_LINT catkin_lint REQUIRED)
    execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
    if(NOT ${lint_result} EQUAL 0)
      message(FATAL_ERROR "catkin_lint failed")
    endif()
endif()
