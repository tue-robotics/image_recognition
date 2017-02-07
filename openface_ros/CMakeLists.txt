cmake_minimum_required(VERSION 2.8.3)
project(openface_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_recognition_msgs
  image_recognition_util
  rospy
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/face_recognition_node
  scripts/get_face_recognition
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)