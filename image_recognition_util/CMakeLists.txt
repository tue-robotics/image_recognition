cmake_minimum_required(VERSION 3.0.2)
project(image_recognition_util)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/publish_image_with_recognitions_overlay

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if (CATKIN_ENABLE_TESTING)
    # Test catkin lint
    find_program(CATKIN_LINT catkin_lint REQUIRED)
    execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
    if(NOT ${lint_result} EQUAL 0)
      message(FATAL_ERROR "catkin_lint failed")
    endif()

    catkin_add_nosetests(test/test_classification_score_matrix.py)
endif()
