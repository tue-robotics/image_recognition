# Color extractor image recognition

Image recognition with use of the color extractor.

## Installation

See https://github.com/tue-robotics/image_recognition for installation instructions.

## ROS Node (color_extractor_node)

Color extraction from https://github.com/algolia/color-extractor.
```
rosrun image_recognition_color_extractor color_extractor_node
```

Run the image_recognition_rqt test gui (https://github.com/tue-robotics/image_recognition_rqt)

    rosrun image_recognition_rqt test_gui

Configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image around a face:

![Color extractor](doc/color_extractor_test.png)

## Scripts

### Get colors (get_colors)

Get the color classification result of an input image:

```
rosrun image_recognition_color_extractor get_colors `rospack find image_recognition_color_extractor`/doc/shirt.png
```

![Example](doc/shirt.png)

Output:

    TODO