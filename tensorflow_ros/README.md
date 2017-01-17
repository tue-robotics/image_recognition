# tensorflow_ros
Object recognition with use of Tensorflow. Based on the retrain example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

RQT tools related to this package can be found in package [tensorflow_ros_rqt](https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt)

## Installation

See https://github.com/tue-robotics/image_recognition for installation instructions. Also make sure that you have a working installation on Tensorflow. If you are running Ubuntu x64, Tensorflow can be installed with use of the following commands:

    sudo apt-get install python-pip python-dev
    sudo pip install tensorflow # If you want gpu support, use pip install tensorflow-gpu

## How-to

1. Annotate images with use of annotation tool provided in https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt
2. Retrain the neural network: https://github.com/tue-robotics/image_recognition/tree/master/tensorflow_ros_rqt
3. Start the ROS node with the net as parameter:

    ```
    rosrun tensorflow_ros object_recognition_node.py _database_path:=[path_to_db]
    ```

4. Test the classifier with use the test tool in https://github.com/tue-robotics/image_recognition/tree/master/image_recognition_rqt
