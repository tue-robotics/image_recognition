cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_analysis)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/classifier_metrics
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
