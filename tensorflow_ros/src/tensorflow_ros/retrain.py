#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from argparse import Namespace

import errno
import tensorflow as tf
from tensorflow_ros.tf_retrain import main as tf_main

defaults = Namespace(steps=1000, batch=100)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main(image_dir, model_dir, output_dir, steps, batch):
    tf.app.flags.FLAGS.image_dir = image_dir
    tf.app.flags.FLAGS.model_dir = model_dir

    mkdir_p(output_dir)
    tf.app.flags.FLAGS.output_graph = os.path.join(output_dir, 'output_graph.pb')
    tf.app.flags.FLAGS.output_labels = os.path.join(output_dir, 'output_labels.txt')

    tf.app.flags.FLAGS.how_many_training_steps = steps

    tf.app.flags.FLAGS.test_batch_size = batch
    tf.app.flags.FLAGS.train_batch_size = batch
    tf.app.flags.FLAGS.validation_batch_size = batch

    tf_main('')
