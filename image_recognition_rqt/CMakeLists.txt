cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_rqt)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  rqt_gui
  rqt_gui_py
  image_recognition_msgs
  image_recognition_util
  sensor_msgs
)

catkin_python_setup()

catkin_package()

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  scripts/annotation_gui
  scripts/manual_gui
  scripts/test_gui
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)