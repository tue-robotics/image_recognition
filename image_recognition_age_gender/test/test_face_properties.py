#!/usr/bin/env python

import os
import re
from future.moves.urllib.request import urlretrieve
import unittest

import cv2
import rospkg
from image_recognition_age_gender.age_gender_estimator import AgeGenderEstimator


@unittest.skip
def test_face_properties():
    local_path = "/tmp/best-epoch47-0.9314.onnx"

    if not os.path.exists(local_path):
        http_path = "https://github.com/Nebula4869/PyTorch-gender-age-estimation/raw/" \
                    "038331d26fc1fbf24d00365d0eb9d0e5e828dda6/models-2020-11-20-14-37/best-epoch47-0.9314.onnx"
        urlretrieve(http_path, local_path)
        print("Downloaded weights to {}".format(local_path))

    def age_is_female_from_asset_name(asset_name):
        age_str, gender_str = re.search("age_(\d+)_gender_(\w+)", asset_name).groups()
        return int(age_str), gender_str == "female"

    assets_path = os.path.join(rospkg.RosPack().get_path("image_recognition_age_gender"), 'test/assets')
    images_gt = [(cv2.imread(os.path.join(assets_path, asset)), age_is_female_from_asset_name(asset))
                 for asset in os.listdir(assets_path)]

    estimations = AgeGenderEstimator(local_path, 64).estimate(image for image, _ in images_gt)
    for (_, (age_gt, is_female_gt)), (age, gender) in zip(images_gt, estimations):
        age = int(age)
        is_female = gender[0] > 0.5
        assert abs(age - age_gt) < 5, f"{age=}, {age_gt=}"
        assert is_female == is_female_gt, f"{is_female=}, {is_female_gt=}"


if __name__ == "__main__":
    test_face_properties()
