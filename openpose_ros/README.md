openpose_ros
====================================

ROS Wrapper for openpose https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Description
Provides a service interface for openpose. Returns the skeleton when an image is send

## Installation
ROS Kinetic uses OpenCV 3.2 as default. Therefore it is important to compile openpose and caffe against OpenCV 3.2 as well

#### A simple example using the opencv 3.2 version distributed by ROS kinetic:
```
sudo apt remove opencv* libopencv*
sudo apt install ros-kinetic-opencv3
ln -s /opt/ros/kinetic/lib/opencv_core3.so /usr/lib/opencv_core.so
ln -s /opt/ros/kinetic/lib/opencv_highgui3.so /usr/lib/opencv_highgui.so
ln -s /opt/ros/kinetic/lib/opencv_imgcodecs3.so /usr/lib/opencv_imgcodecs.so
ln -s /opt/ros/kinetic/lib/opencv_imgproc3.so /usr/lib/opencv_imgproc.so
ln -s /opt/ros/kinetic/lib/opencv_videoio3.so /usr/lib/opencv_videoio.so
```

Next compile openpose using the [openpose installation manual](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md)

Make sure at the end a symlink is added to the ros package, for example if the openpose folder is in your home dir:
```
roscd openpose_ros
ln -s ~/openpose
```

If the symlink is not present a mock node will be used for testing. 

(After creating the symlink, do not forget to clean first)
