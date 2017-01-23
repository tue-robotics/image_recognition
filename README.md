# TU/e Robotics image_recognition
Packages for image recognition - Robocup TU/e Robotics

## Package status & Description

Package | Build status Xenial Kinetic x64 | Description
------- | ------------------------------- | -----------
[image_recognition](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition) | [![Build Status](http://build.ros.org/job/Ksrc_uX__image_recognition__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__image_recognition__ubuntu_xenial__source/1/) | Meta package for all image_recognition packages.
[image_recognition_util](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_util) | [![Build Status](http://build.ros.org/job/Ksrc_uX__image_recognition_util__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__image_recognition_util__ubuntu_xenial__source/1/) | Utils shared among image recognition packages
[image_recognition_msgs](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_msgs) | [![Build Status](http://build.ros.org/job/Ksrc_uX__image_recognition_msgs__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__image_recognition_msgs__ubuntu_xenial__source/1/) | Interface definition for image recognition
[image_recognition_rqt](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt) | [![Build Status](http://build.ros.org/job/Ksrc_uX__image_recognition_rqt__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__image_recognition_rqt__ubuntu_xenial__source/1/) | RQT tools with helpers testing this interface and training/labeling data.
[tensorflow_ros](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros) | [![Build Status](http://build.ros.org/job/Ksrc_uX__tensorflow_ros__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__tensorflow_ros__ubuntu_xenial__source/1/) | Object recognition with use of Tensorflow. The user can retrain the top layers of a neural network to perform classification with its own dataset as described in [this tutorial](https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html).
[tensorflow_ros_rqt](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt) | [![Build Status](http://build.ros.org/job/Ksrc_uX__tensorflow_ros_rqt__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__tensorflow_ros_rqt__ubuntu_xenial__source/1/) | RQT tools for retraining a Tensorflow neural network.
[openface_ros](https://github.com/tue-robotics/image_recognition/tree/master/openface_ros) | [![Build Status](http://build.ros.org/job/Ksrc_uX__openface_ros__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__openface_ros__ubuntu_xenial__source/1/) | ROS wrapper for Openface (https://github.com/cmusatyalab/openface) to detect and recognize faces in images.
[skybiometry_ros](https://github.com/tue-robotics/image_recognition/tree/master/skybiometry_ros) | [![Build Status](http://build.ros.org/job/Ksrc_uX__skybiometry_ros__ubuntu_xenial__source/1//badge/icon)](http://build.ros.org/job/Ksrc_uX__skybiometry_ros_ubuntu_xenial__source/1/) | ROS wrapper for Skybiometry (https://skybiometry.com/) for getting face properties of a detected face, e.g. age estimation, gender estimation etc.

## Travis CI Build Status

[![Build Status](https://travis-ci.org/tue-robotics/image_recognition.svg)](https://travis-ci.org/tue-robotics/image_recognition)

# How to

## Object recognition
Step 1: label images with the [image_recognition_rqt#annotation-plugin](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt#annotation-plugin)

[![Annotate](http://img.youtube.com/vi/uAQvn7SInlg/0.jpg)](http://www.youtube.com/watch?v=uAQvn7SInlg)
<-- Youtube video

Step 2: train a neural network with the [tensorflow_ros_rqt](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt)

[![Train](http://img.youtube.com/vi/6JdtWa8FD04/0.jpg)](http://www.youtube.com/watch?v=6JdtWa8FD04)
<-- Youtube video

Step 3: predict labels for new data with the [image_recognition_rqt#test-plugin](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt#test-plugin)

[![Recognize](http://img.youtube.com/vi/OJKYLB3myWw/0.jpg)](http://www.youtube.com/watch?v=OJKYLB3myWw)
<-- Youtube video

## Face recognition
See the tutorial at [openface_ros](https://github.com/tue-robotics/image_recognition/tree/master/openface_ros)

[![Face recognition](http://img.youtube.com/vi/yGqDdfYxHZw/0.jpg)](http://www.youtube.com/watch?v=yGqDdfYxHZw)
<-- Youtube video

# Installation

Clone the repo in your catkin_ws:
        
        cd ~/catkin_ws/src
        git clone https://github.com/tue-robotics/image_recognition.git
        
Build your catkin workspace
        cd ~/catkin_ws
        catkin_make
