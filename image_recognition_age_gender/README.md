# Image recognition pytorch

Image recognition (age and gender estimation of a face) with use of PyTorch.

## Installation

See https://github.com/tue-robotics/image_recognition for installation instructions.

## ROS Node (face_properties_node)

Age and gender estimation
```
rosrun image_recognition_pytorch face_properties_node _weights_file_path:=[path_to_model]
```

Run the image_recognition_rqt test gui (https://github.com/tue-robotics/image_recognition_rqt)

    rosrun image_recognition_rqt test_gui

Configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image around a face:

![Wide ResNet](doc/wide_resnet_test.png)

## Scripts

### Download model

Download weights from github.

```
usage: download_model [-h] [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
```

### Get face properties (get_face_properties)

Get the classification result of an input image:

```
rosrun image_recognition_pytorch get_face_properties `rospack find image_recognition_pytorch`/doc/face.png
```

![Example](doc/face.png)

Output:

    [(50.5418073660112, array([0.5845756 , 0.41542447], dtype=float32))]
