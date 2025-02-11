cmake_minimum_required(VERSION 3.5)
project(image_recognition_pose_estimation)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/detect_poses
  scripts/pose_estimation_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test("-W2")

  catkin_add_nosetests(test)
endif()
