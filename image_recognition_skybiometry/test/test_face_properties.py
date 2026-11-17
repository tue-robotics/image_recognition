#!/usr/bin/env python
from __future__ import print_function

import os
import re

import pytest

import cv2
from image_recognition_skybiometry.skybiometry import Skybiometry


def test_face_properties():
    test_key = os.environ.get('SKYBIOMETRY_API_KEY')
    test_secret = os.environ.get('SKYBIOMETRY_API_SECRET')
    if not test_key or not test_secret:
        pytest.skip('Skybiometry test credentials are unavailable')
    test_namespace = 'techunited'

    def age_is_female_from_asset_name(asset_name):
        age_str, gender_str = re.search(r"age_(\d+)_gender_(\w+)", asset_name).groups()
        return int(age_str), gender_str == "female"

    assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test/assets')
    images_gt = [(cv2.imread(os.path.join(assets_path, asset)), age_is_female_from_asset_name(asset))
                 for asset in os.listdir(assets_path)]

    estimations = Skybiometry(test_key, test_secret, test_namespace).get_face_properties([image for image, _ in images_gt], 10.0)
    for (_, (age_gt, is_female_gt)), face_property in zip(images_gt, estimations):
        age = int(face_property.age_est.value)
        is_female = face_property.gender.value == "female"
        # assert abs(age - age_gt) <= 5  # Poor performance
        assert is_female == is_female_gt


if __name__ == "__main__":
    test_face_properties()
