#!/usr/bin/env python

import os

import rospkg


def test_retrain_and_evaluate():
    # Direct call to script because rosrun and packages could not yet be available in environment
    pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    models_path = os.path.join(pkg_path, 'test/models')
    assets_path = os.path.join(pkg_path, 'test/assets')
    assert 0 == os.system("{} {} {} {}".format(os.path.join(pkg_path, "scripts/retrain"),
                                               os.path.join(assets_path, 'training'),
                                               models_path,
                                               models_path))
    assert 0 == os.system("{} {} {} {}".format(os.path.join(pkg_path, "scripts/evaluate_classifier"),
                                               os.path.join(models_path, 'output_graph.pb'),
                                               os.path.join(models_path,
                                                            'output_labels.txt'),
                                               os.path.join(assets_path, 'verification')))
