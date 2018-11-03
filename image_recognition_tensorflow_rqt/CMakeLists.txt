cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_tensorflow_rqt)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  scripts/train_gui
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)