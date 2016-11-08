# openface_ros

Face recognition with use of Openface (https://cmusatyalab.github.io/openface/)

## Installation

See https://github.com/tue-robotics/image_recognition

## How-to

### ROS Node



### Command line

Command line interface to test the detection / recognition based on an image:

    usage: get_face_recognition.py [-h] -i IMAGE [-k ALIGN_PATH] [-s NET_PATH] [-v]

Run the command on an example image:

    rosrun openface_ros get_face_recognition.py -i `rospack find openface_ros`/doc/example.png

This will lookup this image in the openface_ros/doc folder and perform recognitions

![Example](doc/example.png)

Output: 

    [RecognizedFace(roi=(374, 188, 108, 123), l2_distances=[]), RecognizedFace(roi=(72, 147, 88, 105), l2_distances=[]), RecognizedFace(roi=(377, 95, 74, 86), l2_distances=[]), RecognizedFace(roi=(149, 26, 74, 86), l2_distances=[]), RecognizedFace(roi=(52, 47, 75, 86), l2_distances=[]), RecognizedFace(roi=(246, 115, 88, 102), l2_distances=[]), RecognizedFace(roi=(0, 0, 42, 60), l2_distances=[]), RecognizedFace(roi=(336, 33, 74, 86), l2_distances=[]), RecognizedFace(roi=(228, 0, 62, 60), l2_distances=[])]

Since no faces were trained, the l2_distances will not be calculated of-course.
    
