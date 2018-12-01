cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_openpose)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/detect_poses
  scripts/openpose_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
