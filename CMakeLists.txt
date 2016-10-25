cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_object_recognition)

find_package(catkin REQUIRED COMPONENTS
  rospy
  object_recognition_srvs
)

catkin_python_setup()

catkin_package()
