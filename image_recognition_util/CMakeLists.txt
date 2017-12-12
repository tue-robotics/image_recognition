cmake_minimum_required(VERSION 2.8.3)
project(image_recognition_util)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

if (CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    add_rostest(test/test_all.test)
endif()
