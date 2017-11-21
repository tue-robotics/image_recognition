cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_tools)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  python-matplotlib
  python-scipy
  python-sklearn
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/classifier_metric
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)