cmake_minimum_required(VERSION 3.0.2)
project(image_recognition_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  sensor_msgs
)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  FaceProperties.msg
  CategoryProbability.msg
  CategoricalDistribution.msg
  Recognition.msg
  Recognitions.msg
  Annotation.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  Recognize.srv
  Annotate.srv
  GetFaceProperties.srv
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test()
endif()
