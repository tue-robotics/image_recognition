cmake_minimum_required(VERSION 3.5)
project(image_recognition_openpose)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/detect_poses
  scripts/openpose_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test()
endif()
