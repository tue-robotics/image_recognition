# image_recognition_segment_anything

ROS Wrapper around the segment anything model

## Installation

See <https://github.com/tue-robotics/image_recognition> for installation instructions.

## ROS Node (footwear_node)

```bash
rosrun image_recognition_segment_anything footwear_node
```

Run the image_recognition_rqt test gui (<https://github.com/tue-robotics/image_recognition_rqt>)

```bash
rosrun image_recognition_rqt test_gui
```

Configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image around the region you would like to extract the color from.

## Scripts

### Get footwear (get_footwear)

Get whether there are shoes or socks in the image:

```bash
rosrun image_recognition_segment_anything get_footwear `rospack find image_recognition_footwear`/doc/shoes.png
rosrun image_recognition_segment_anything get_footwear `rospack find image_recognition_footwear`/doc/socks.png
```

![Shoes](doc/shoes.png)
![Socks](doc/socks.png)

Output:

```yaml
Colors: ['light blue']  # ToDo adapt
```
