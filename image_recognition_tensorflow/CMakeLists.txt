cmake_minimum_required(VERSION 3.8)
project(image_recognition_tensorflow)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(PROGRAMS
  scripts/evaluate_classifier
  scripts/get_object_recognition
  scripts/object_recognition_node
  scripts/retrain
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(python_tests test/test_retrain_and_evaluate.py)
endif()

ament_package()
