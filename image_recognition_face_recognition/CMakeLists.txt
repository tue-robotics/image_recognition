cmake_minimum_required(VERSION 3.0.2)
project(image_recognition_face_recognition)


find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/face_recognition_node
  scripts/get_face_recognition
  scripts/train_from_images
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test("-W2")

  catkin_add_nosetests(test)
endif()
