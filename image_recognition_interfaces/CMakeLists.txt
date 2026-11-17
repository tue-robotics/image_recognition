cmake_minimum_required(VERSION 3.8)
project(image_recognition_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Annotation.msg"
  "msg/CategoricalDistribution.msg"
  "msg/CategoryProbability.msg"
  "msg/FaceProperties.msg"
  "msg/Recognition.msg"
  "msg/Recognitions.msg"
  "srv/Annotate.srv"
  "srv/GetFaceProperties.srv"
  "srv/Recognize.srv"
  DEPENDENCIES sensor_msgs std_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)
  ament_lint_cmake(MAX_LINE_LENGTH 120)
  ament_xmllint(MAX_LINE_LENGTH 120)
endif()

ament_package()
