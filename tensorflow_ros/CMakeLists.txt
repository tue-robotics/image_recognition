cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  image_recognition_msgs
  image_recognition_util
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/retrain
  scripts/object_recognition_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)