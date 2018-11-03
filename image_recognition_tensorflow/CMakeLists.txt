cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_tensorflow)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/evaluate_classifier
  scripts/get_object_recognition
  scripts/object_recognition_node
  scripts/retrain
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)