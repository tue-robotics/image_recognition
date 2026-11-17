cmake_minimum_required(VERSION 3.8)
project(image_recognition_openface)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(PROGRAMS
  scripts/face_recognition_node
  scripts/get_face_recognition
  scripts/train_from_images
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
