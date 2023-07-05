#!/usr/bin/env python

import os
import re
from future.moves.urllib.request import urlretrieve
import unittest

from PIL import Image
import rospkg
from image_recognition_footwear.model import Model
from image_recognition_footwear.process_data import heroPreprocess, detection_RGB
import torch

@unittest.skip
def test_footwear():
    local_path = "~/data/pytorch_models/footwearModel.pth"

    if not os.path.exists(local_path):
        print("File does not exit {}".format(local_path))

    def is_there_footwear_from_asset_name(asset_name):
        binary_str = re.search("(\w+)_shoe", asset_name).groups()
        return binary_str == "yes"

    assets_path = os.path.join(rospkg.RosPack().get_path("image_recognition_footwear"), 'test/assets')
    images_gt = [(Image.open(os.path.join(assets_path, asset)), is_there_footwear_from_asset_name(asset))
                 for asset in os.listdir(assets_path)]

    device = torch.device('cuda')
    model = Model(in_channel=3, channel_1=128, channel_2=256, channel_3=512, node_1=1024, node_2=1024, num_classes=2)
    model.load_state_dict(torch.load(local_path))
    model.to(device=device)
    detections = detection_RGB([image for image, _ in images_gt], model)

    estimations = AgeGenderEstimator(local_path, 64, 16, 8).estimate([image for image, _ in images_gt])

    for (_, (is_footwear_gt)), (binary_detection) in zip(images_gt, detections):
        binary_detection = int(binary_detection)
        assert is_footwear_gt == binary_detection, f"{binary_detection=}, {is_footwear_gt=}"


if __name__ == "__main__":
    test_footwear()