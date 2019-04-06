# Openpose image recognition

ROS Wrapper for openpose https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Installation notes

This ROS wrapper makes use of the [Openpose python interface](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md).
Please follow the [installation manual](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md) and ensure that the `BUILD_PYTHON` flag is turned on while running CMake. Also make sure that you install a release instead of the latest master version with **CUDA8** since Torch (image_recognition_openface) cannot handle **CUDA10**.

```bash
export OPENPOSE_INSTALL_PATH=~/openpose && \
mkdir -p $OPENPOSE_INSTALL_PATH && \
wget https://github.com/CMU-Perceptual-Computing-Lab/openpose/archive/v1.4.0.tar.gz -O /tmp/v1.4.0.tar.gz && \
tar -xvf /tmp/v1.4.0.tar.gz -C /tmp/ && cp -r /tmp/openpose-1.4.0/* $OPENPOSE_INSTALL_PATH  && \
$OPENPOSE_INSTALL_PATH/ubuntu/install_cuda.sh && \
$OPENPOSE_INSTALL_PATH/ubuntu/install_cudnn.sh && \
wget https://github.com/CMU-Perceptual-Computing-Lab/caffe/archive/1.0.tar.gz -O /tmp/1.0.tar.gz && \
tar -xvf /tmp/1.0.tar.gz -C /tmp/ && cp -r /tmp/caffe-1.0/* $OPENPOSE_INSTALL_PATH/3rdparty/caffe && \
cd $OPENPOSE_INSTALL_PATH && mkdir -p build && cd build && \
cmake .. -DBUILD_PYTHON=1 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda && \
make -j`nproc` && \
sudo make install
```

## Scripts

### detect_poses

Example for the following picture:

![Example](doc/example.jpg)

```bash
rosrun image_recognition_openpose detect_poses image `rospack find image_recognition_openpose`/doc/example.jpg
```

Output:

![Example result](doc/example_result.jpg)

It also works with a webcam stream, usage:

```bash
usage: detect_poses [-h] [--pose_model POSE_MODEL]
                    [--net_input_size NET_INPUT_SIZE]
                    [--net_output_size NET_OUTPUT_SIZE]
                    [--num_scales NUM_SCALES] [--scale_gap SCALE_GAP]
                    [--num_gpu_start NUM_GPU_START]
                    [--overlay_alpha OVERLAY_ALPHA]
                    [--python_path PYTHON_PATH]
                    model_folder {image,cam} ...

Detect poses in an image

positional arguments:
  model_folder          Path where the models are stored
  {image,cam}           Mode
    image               Use image mode
    cam                 Use cam mode

optional arguments:
  -h, --help            show this help message and exit
  --pose_model POSE_MODEL
                        What pose model to use (default: BODY_25)
  --net_input_size NET_INPUT_SIZE
                        Net input size (default: -1x368)
  --net_output_size NET_OUTPUT_SIZE
                        Net output size (default: -1x-1)
  --num_scales NUM_SCALES
                        Num scales (default: 1)
  --scale_gap SCALE_GAP
                        Scale gap (default: 0.3)
  --num_gpu_start NUM_GPU_START
                        What GPU support (default: 0)
  --overlay_alpha OVERLAY_ALPHA
                        Overlay alpha for the output image (default: 0.6)
  --python_path PYTHON_PATH
                        Python path where Openpose is stored (default:
                        /usr/local/python/)
```

### openpose_node

## How-to

Run the image_recognition_openpose node in one terminal, e.g.:

```bash
rosrun image_recognition_openpose openpose_node
```

Next step is starting the image_recognition_Rqt test gui (https://github.com/tue-robotics/image_recognition_rqt)

    rosrun image_recognition_rqt test_gui

Configure the service you want to call with the gear-wheel in the top-right corner of the screen. If everything is set-up, draw a rectangle in the image and ask the service for detections:

![Test](doc/openpose.png)

You will see that the result of the detection will prompt in a dialog combo box. Also the detections will be drawn on the image. The ROS node also published the result image, you can easily view this image using `rqt_image_view`.
