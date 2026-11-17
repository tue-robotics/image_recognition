cmake_minimum_required(VERSION 3.8)
project(image_recognition_age_gender)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(PROGRAMS
  scripts/download_model
  scripts/face_properties_node
  scripts/get_face_properties
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(face_properties test/test_face_properties.py)
endif()

ament_package()
