jetson_inference_ros
====================================

ROS Wrapper for Jetson inference https://github.com/dusty-nv/jetson-inference

## Description
Provides a service and topic interface for jetson inference.

![Illustration](doc/illustration.png)

## Installation on Jetson TX2

Run the install jetson-inference script

    rosrun jetson_inference_ros install_jetson_inference.bash 

If the jetson-inference cannot be found using CMake, it will compile a mock.

