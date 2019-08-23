# image_recognition_color_extractor

Package to extractor strongest colors from an image

## Installation

See https://github.com/tue-robotics/image_recognition for installation instructions.

## ROS Node (color_extractor_node)

```
rosrun image_recognition_color_extractor color_extractor_node _total:=3
```

Run the image_recognition_rqt test gui (https://github.com/tue-robotics/image_recognition_rqt)

    rosrun image_recognition_rqt test_gui

Configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image around the region you would like to extract the color from.

## Scripts

### Get colors (get_colors)

Get the most dominant colors from an image:

```
rosrun image_recognition_color_extractor get_colors `rospack find image_recognition_color_extractor`/doc/example.png
```

![Example](doc/example.png)

Output:

```
Colors: ['light blue']
```
