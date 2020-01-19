#!/usr/bin/env python
from __future__ import print_function

import os
import re
import urllib

import cv2
import rospkg
from image_recognition_keras.age_gender_estimator import AgeGenderEstimator


def test_face_properties():
    local_path = "/tmp/age_gender_weights.hdf5"

    if not os.path.exists(local_path):
        http_path = "https://github.com/tue-robotics/image_recognition/releases/download/" \
                    "image_recognition_keras_face_properties_weights.28-3.73/" \
                    "image_recognition_keras_face_properties_weights.28-3.73.hdf5"
        urllib.urlretrieve(http_path, local_path)
        print("Downloaded weights to {}".format(local_path))

    def age_is_female_from_asset_name(asset_name):
        age_str, gender_str = re.search("age_(\d+)_gender_(\w+)", asset_name).groups()
        return int(age_str), gender_str == "female"

    assets_path = os.path.join(rospkg.RosPack().get_path("image_recognition_keras"), 'test/assets')
    images_gt = [(cv2.imread(os.path.join(assets_path, asset)), age_is_female_from_asset_name(asset))
                 for asset in os.listdir(assets_path)]

    estimations = AgeGenderEstimator(local_path, 64, 16, 8).estimate([image for image, _ in images_gt])
    for (_, (age_gt, is_female_gt)), (age, gender) in zip(images_gt, estimations):
        age = int(age)
        is_female = gender[0] > 0.5
        assert abs(age - age_gt) < 5
        assert is_female == is_female_gt


if __name__ == "__main__":
    test_face_properties()
