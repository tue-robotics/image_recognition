# TU/e Robotics image_recognition

[![CI](https://github.com/tue-robotics/image_recognition/actions/workflows/main.yml/badge.svg)](https://github.com/tue-robotics/image_recognition/actions/workflows/main.yml)
[![Industrial CI](https://github.com/tue-robotics/image_recognition/actions/workflows/industrial_ci.yml/badge.svg)](https://github.com/tue-robotics/image_recognition/actions/workflows/industrial_ci.yml)

Packages for image recognition - Robocup TU/e Robotics

## ROS 2 packages

The packages target ROS 2 Jazzy. The [interface package](image_recognition_interfaces)
defines the messages and services shared by the recognition nodes, tools, and
[graphical plugins](image_recognition_rqt). The [grouping package](image_recognition)
lists dependencies for the main collection.

# How to

## Object recognition
Step 1: label images with the [image_recognition_rqt#annotation-plugin](image_recognition_rqt#annotation-plugin)

[![Annotate](http://img.youtube.com/vi/uAQvn7SInlg/0.jpg)](http://www.youtube.com/watch?v=uAQvn7SInlg)
<-- Youtube video

Step 2: train a neural network with the [image_recognition_tensorflow_rqt](image_recognition_tensorflow_rqt)

[![Train](http://img.youtube.com/vi/6JdtWa8FD04/0.jpg)](http://www.youtube.com/watch?v=6JdtWa8FD04)
<-- Youtube video

Step 3: predict labels for new data with the [image_recognition_rqt#test-plugin](image_recognition_rqt#test-plugin)

[![Recognize](http://img.youtube.com/vi/OJKYLB3myWw/0.jpg)](http://www.youtube.com/watch?v=OJKYLB3myWw)
<-- Youtube video

## Face recognition
See the tutorial at [image_recognition_openface](image_recognition_openface)

[![Face recognition](http://img.youtube.com/vi/yGqDdfYxHZw/0.jpg)](http://www.youtube.com/watch?v=yGqDdfYxHZw)
<-- Youtube video

# Building locally

From a ROS 2 workspace containing the repository, with the dependencies for the
packages you intend to use installed:

```bash
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

For a small build that does not require the recognition models:

```bash
colcon build --packages-up-to image_recognition_util
source install/setup.bash
ros2 interface show image_recognition_interfaces/srv/Recognize
```
