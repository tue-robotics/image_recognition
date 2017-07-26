cmake_minimum_required(VERSION 2.8.3)
project(openpose_ros)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  sensor_msgs
  image_recognition_msgs
)

catkin_package()

set(OPENPOSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/openpose)
IF(EXISTS ${OPENPOSE_DIR})
  message("Using openpose from " ${OPENPOSE_DIR} " ...")

  set(CAFFE_DIR ${OPENPOSE_DIR}/3rdparty/caffe)

  set(CAFFE_INCLUDE_DIRS ${CAFFE_DIR}/distribute/include)
  set(CAFFE_LINK_LIBRARY ${CAFFE_DIR}/distribute/lib/libcaffe.so)

  set(OPENPOSE_INCLUDE_DIRS ${OPENPOSE_DIR}/include openpose/src /usr/local/cuda/include)
  set(OPENPOSE_LINK_LIBRARY ${OPENPOSE_DIR}/build/lib/libopenpose.so)

  add_definitions(-DUSE_CAFFE)

  include_directories(
    ${catkin_INCLUDE_DIRS}
    ${CAFFE_INCLUDE_DIRS}
    ${OPENPOSE_INCLUDE_DIRS}
  )

  add_library(openpose_wrapper src/openpose_wrapper.cpp)

  target_link_libraries(openpose_wrapper 
    ${CAFFE_LINK_LIBRARY}
    ${OPENPOSE_LINK_LIBRARY}
    ${catkin_LIBRARIES}
  )

  add_executable(${PROJECT_NAME}_test test/test.cpp)
  target_link_libraries(${PROJECT_NAME}_test openpose_wrapper)
else()
  message(WARNING "Openpose folder not found in " ${OPENPOSE_DIR} " using a mock instead ...")

  include_directories(${catkin_INCLUDE_DIRS})
  add_library(openpose_wrapper src/openpose_wrapper_mock.cpp)

  target_link_libraries(openpose_wrapper ${catkin_LIBRARIES})
endif()

# depend on generated messages
add_dependencies(openpose_wrapper ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_node src/openpose_ros_node.cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node openpose_wrapper)
