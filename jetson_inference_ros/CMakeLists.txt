cmake_minimum_required(VERSION 2.8.3)
project(jetson_inference_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_recognition_msgs
  image_transport
  roscpp
  sensor_msgs
)

find_package(jetson-inference REQUIRED)
find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

catkin_package()

add_executable(detect_net_ros src/detect_net_ros.cpp)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(detect_net_ros
  ${catkin_LIBRARIES}
  jetson-inference
)

