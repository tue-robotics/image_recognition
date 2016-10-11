cmake_minimum_required(VERSION 2.8.3)
project(object_recognition_rqt)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  rqt_gui
  rqt_gui_py
)

catkin_python_setup()

catkin_package()

install(PROGRAMS scripts/object_recognition_rqt
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

