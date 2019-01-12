#!/usr/bin/env python

import os

import rospkg

pkg_name = 'image_recognition_tensorflow'
pkg_path = rospkg.RosPack().get_path(pkg_name)
models_path = os.path.join(pkg_path, 'test/models')
assets_path = os.path.join(pkg_path, 'test/assets')


def test_retrain_and_evaluate():
    assert 0 == os.system("rosrun {} retrain {} {} {}".format(pkg_name,
                                                              os.path.join(assets_path, 'training'),
                                                              models_path,
                                                              models_path))
    assert 0 == os.system("rosrun {} evaluate_classifier {} {} {}".format(pkg_name,
                                                                          os.path.join(models_path, 'output_graph.pb'),
                                                                          os.path.join(models_path,
                                                                                       'output_labels.txt'),
                                                                          os.path.join(assets_path, 'verification')))
