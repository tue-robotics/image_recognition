#!/usr/bin/env python

import os
import unittest

import rospkg
import rostest


class TestRetrainAndEvaluate(unittest.TestCase):
    def test_retrain_and_evaluate(self):
        pkg_name = 'image_recognition_tensorflow'
        pkg_path = rospkg.RosPack().get_path(pkg_name)
        models_path = os.path.join(pkg_path, 'test/models')
        assets_path = os.path.join(pkg_path, 'test/assets')
        self.assertFalse(os.system("rosrun {} retrain {} {} {}".format(pkg_name,
                                                                       os.path.join(assets_path, 'training'),
                                                                       models_path,
                                                                       models_path)))
        self.assertFalse(os.system("rosrun {} evaluate_classifier {} {} {}".format(pkg_name,
                                                                                   os.path.join(models_path,
                                                                                                'output_graph.pb'),
                                                                                   os.path.join(models_path,
                                                                                                'output_labels.txt'),
                                                                                   os.path.join(assets_path,
                                                                                                'verification'))))


if __name__ == '__main__':
    rostest.rosrun("image_recognition_tensorflow", 'test_all', TestRetrainAndEvaluate)
