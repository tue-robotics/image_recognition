cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_analysis)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/classifier_metrics
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if (CATKIN_ENABLE_TESTING)
    # Test catkin lint
    find_program(CATKIN_LINT catkin_lint REQUIRED)
    execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
    if(NOT ${lint_result} EQUAL 0)
      message(FATAL_ERROR "catkin_lint failed")
    endif()
endif()
