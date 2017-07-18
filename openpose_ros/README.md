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

sudo ln -fs /opt/ros/kinetic/lib/libopencv_core3.so /usr/lib/libopencv_core.so
sudo ln -fs /opt/ros/kinetic/lib/libopencv_highgui3.so /usr/lib/libopencv_highgui.so
sudo ln -fs /opt/ros/kinetic/lib/libopencv_imgcodecs3.so /usr/lib/libopencv_imgcodecs.so
sudo ln -fs /opt/ros/kinetic/lib/libopencv_imgproc3.so /usr/lib/libopencv_imgproc.so
sudo ln -fs /opt/ros/kinetic/lib/libopencv_videoio3.so /usr/lib/libopencv_videoio.so
sudo ln -fs /opt/ros/kinetic/include/opencv-3.2.0-dev/opencv2 /usr/include/opencv2
```

Next compile openpose using the [openpose installation manual](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md)

Make sure at the end a symbolic link is added to the ROS package, for example if the openpose folder is in your home dir:
```
roscd openpose_ros
ln -s ~/openpose
```

If the symbolic link is not present a mock node will be used for testing. 

(After creating the symlink, do not forget to clean first)

## How-to

Run the openpose_ros node in one terminal, e.g.:

    rosrun openpose_ros openpose_ros_node _net_input_width:=368 _net_input_height:=368 _net_output_width:=368 _net_output_height:=368 _model_folder:=/home/ubuntu/openpose/models/

Next step is starting the image_recognition_Rqt test gui (https://github.com/tue-robotics/image_recognition_rqt)

    rosrun image_recognition_rqt test_gui
    
Again configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image and ask the service for detections:

![Test](doc/openpose.png)

You will see that the result of the detection will prompt in a dialog combo box. Also the detections will be drawn on the image. The ROS node also published the result image, you can easily view this image using `rqt_image_view`.
