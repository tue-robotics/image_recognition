# TU/e Robotics image_recognition
Packages for image recognition - Robocup TU/e Robotics

# How to

## Object recognition
Step 1: label images with the [image_recognition_rqt#annotation-plugin](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt#annotation-plugin)

Step 2: train a neural network with the [tensorflow_ros_rqt](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt)

Step 3: predict labels for new data with the [image_recognition_rqt#test-plugin](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt#test-plugin)

## Face recognition
See the tutorial at [openface_ros](https://github.com/tue-robotics/image_recognition/tree/master/openface_ros)

# Installation

Clone the repo in your catkin_ws:
        
        cd ~/catkin_ws/src
        git clone https://github.com/tue-robotics/image_recognition.git
        
Build your catkin workspace
        cd ~/catkin_ws
        catkin_make

# Packages

- [image_recognition](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition) - Meta package for all image_recognition packages.
- [image_recognition_util](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_util) - Utils shared among image recognition packages
- [image_recognition_msgs](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_msgs) - Interface definition for image recognition
- [image_recognition_rqt](https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt) - RQT tools with helpers testing this interface and training/labeling data.
- [tensorflow_ros](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros) - Object recognition with use of Tensorflow. The user can retrain the top layers of a neural network to perform classification with its own dataset as described in this tutorial (https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html).
- [tensorflow_ros_rqt](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt) - RQT tools for retraining a Tensorflow neural network.
- [openface_ros](https://github.com/tue-robotics/image_recognition/tree/master/openface_ros) - ROS wrapper for Openface (https://github.com/cmusatyalab/openface) to detect and recognize faces in images.
- [skybiometry_ros](https://github.com/tue-robotics/image_recognition/tree/master/skybiometry_ros) - ROS wrapper for Skybiometry (https://skybiometry.com/) for getting face properties of a detected face, e.g. age estimation, gender estimation etc.