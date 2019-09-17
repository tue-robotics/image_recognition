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

if (CATKIN_ENABLE_TESTING)
    # Test catkin lint
    find_program(CATKIN_LINT catkin_lint REQUIRED)
    execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
    if(NOT ${lint_result} EQUAL 0)
      message(FATAL_ERROR "catkin_lint failed")
    endif()
endif()
