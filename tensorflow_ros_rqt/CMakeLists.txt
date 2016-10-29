cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_ros_rqt)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rqt_gui
  rqt_gui_py
)

catkin_python_setup()

catkin_package()

install(PROGRAMS scripts/tensorflow_ros_rqt
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

