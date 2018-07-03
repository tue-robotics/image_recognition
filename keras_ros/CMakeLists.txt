cmake_minimum_required(VERSION 2.8.3)
project(keras_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  image_recognition_msgs
  image_recognition_util
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/face_properties_node
  scripts/get_face_properties
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

