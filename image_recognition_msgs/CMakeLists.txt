cmake_minimum_required(VERSION 2.8.3)
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
  Annotation.msg
  BodypartDetection.msg
  PersonDetection.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  Recognize.srv
  Annotate.srv
  GetFaceProperties.srv
  GetPersons.srv
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_generation sensor_msgs
)