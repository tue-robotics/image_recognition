cmake_minimum_required(VERSION 2.8.3)
project(skybiometry_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_recognition_msgs
  rospy
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/face_properties_node
  scripts/get_face_properties
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)