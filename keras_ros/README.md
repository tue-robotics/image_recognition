# keras_ros

Image recognition with use of Keras.

## Installation

See https://github.com/tue-robotics/image_recognition for installation instructions. 

## ROS Node (face_properties_node)

Age and gender estimation with use of WideResNet from https://github.com/yu4u/age-gender-estimation. You can download the pre-trained model here: https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5

```
rosrun keras_ros face_properties_node _weight_file_path:=[path_to_model]
```

Run the image_recognition_rqt test gui (https://github.com/tue-robotics/image_recognition_rqt)

    rosrun image_recognition_rqt test_gui
    
Configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image around a face:

![Wide ResNet](doc/wide_resnet_test.png)

## Scripts

### Get face properties (get_face_properties)

Get the classification result of an input image:

```
rosrun keras_ros get_face_properties --image doc/face.png --weights-path [path_to_model]
```

![Example](doc/face.png)

Output: 

    [(50.5418073660112, array([0.5845756 , 0.41542447], dtype=float32))]
