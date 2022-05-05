cmake_minimum_required(VERSION 3.0.2)
project(image_recognition_skybiometry)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/face_properties_node
  scripts/get_face_properties
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test()

  catkin_add_nosetests(test)
endif()
