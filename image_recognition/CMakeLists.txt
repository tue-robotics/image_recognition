cmake_minimum_required(VERSION 2.8.3)
project(image_recognition)
find_package(catkin REQUIRED)
catkin_metapackage()

# Tests

# Catkin lint
find_program(CATKIN_LINT catkin_lint REQUIRED)
execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
if(NOT ${lint_result} EQUAL 0)
  message(FATAL_ERROR "catkin_lint failed")
endif()