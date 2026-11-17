cmake_minimum_required(VERSION 3.8)
project(image_recognition_rqt)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(FILES plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/annotation_gui
  scripts/folder_image_publisher
  scripts/manual_gui
  scripts/test_gui
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
