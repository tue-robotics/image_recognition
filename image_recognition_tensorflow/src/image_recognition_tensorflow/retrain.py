#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import sys
from argparse import Namespace

import tensorflow as tf
from image_recognition_tensorflow.tf_retrain import main as tf_main

defaults = Namespace(steps=1000, batch=100)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main(image_dir, model_dir, output_dir, steps, batch, architecture="inception_v3", flip_left_right=False,
         random_crop=0, random_scale=0, random_brightness=0):
    tf.compat.v1.app.flags.FLAGS.image_dir = image_dir
    tf.compat.v1.app.flags.FLAGS.model_dir = model_dir

    mkdir_p(output_dir)
    tf.compat.v1.app.flags.FLAGS.output_graph = os.path.join(output_dir, 'output_graph.pb')
    tf.compat.v1.app.flags.FLAGS.output_labels = os.path.join(output_dir, 'output_labels.txt')

    tf.compat.v1.app.flags.FLAGS.architecture = architecture

    tf.compat.v1.app.flags.FLAGS.how_many_training_steps = steps

    tf.compat.v1.app.flags.FLAGS.test_batch_size = batch
    tf.compat.v1.app.flags.FLAGS.train_batch_size = batch
    tf.compat.v1.app.flags.FLAGS.validation_batch_size = batch

    tf.compat.v1.app.flags.FLAGS.flip_left_right = flip_left_right
    tf.compat.v1.app.flags.FLAGS.random_crop = random_crop
    tf.compat.v1.app.flags.FLAGS.random_scale = random_scale
    tf.compat.v1.app.flags.FLAGS.random_brightness = random_brightness

    # Do not pass additional flags to tf (from rqt and nose)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("-")]

    tf_main('')
