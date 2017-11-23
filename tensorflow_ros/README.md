# tensorflow_ros
Object recognition with use of Tensorflow. Based on the retrain example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

RQT tools related to this package can be found in package [tensorflow_ros_rqt](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt)

## Installation

See https://github.com/tue-robotics/image_recognition for installation instructions. Also make sure that you have a working installation on Tensorflow. If you are running Ubuntu x64, Tensorflow can be installed with use of the following commands:

    sudo apt-get install python-pip python-dev
    sudo pip install tensorflow # If you want gpu support, use pip install tensorflow-gpu

## Quick How-to

1. Annotate images with use of annotation tool provided in https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt
2. Retrain the neural network: https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt
3. Start the ROS node with the net as parameter:

    ```
    rosrun tensorflow_ros object_recognition_node _graph_path:=[path_to_graph.pb] _labels_path:=[path_to_labels.txt]
    ```

4. Test the classifier with use the test tool in https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt

## ROS Node (object_recognition_node)

```
rosrun tensorflow_ros object_recognition_node _graph_path:=[path_to_graph.pb] _labels_path:=[path_to_labels.txt]
```

## Scripts

### Get object recognition (get_object_recognition)

Get the classification result of an input image:

```
rosrun tensorflow_ros get_object_recognition [path_to_graph.pb] [path_to_labels.txt] [path_to_image]
```

### Evaluate classifier (evaluate_classifier)

Evaluate the classifier based on a structured images folder. The script assumes that the images in this directory are separated in different directories with the label name as directory name.

```
rosrun tensorflow_ros evaluate_classifier [path_to_graph.pb] [path_to_labels.txt] [path_to_image_dir]
```

### Retrain the neural network (retrain)

Train the neural network based on a set of images. The script assumes that the images in this directory are separated in different directories with the label name as directory name.

```
rosrun tensorflow_ros retrain [image_folder] [model_folder_inceptionv3] [output_dir]
```





