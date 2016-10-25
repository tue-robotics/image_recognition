#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import hashlib
import operator
import os.path
import random
import re
import shutil
from argparse import ArgumentParser, Namespace
from datetime import datetime
from itertools import groupby

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.util import compat


defaults = Namespace(steps=4000, batch=100)


# global settings
VALIDATION_PERCENTAGE = 10
BOTTLENECK_DIR = '/tmp/bottleneck',
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
SUMMARIES_DIR = '/tmp/retrain_logs'

# learning settings
LEARNING_RATE = 0.01
EVAL_STEP_INTERVAL = 10

# graph settings
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048

FINAL_TENSOR_NAME = 'final_result'


def create_inception_graph(model_dir):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
    RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with open(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def create_image_lists(image_dir, validation_percentage, validation=False):
    if not os.path.isdir(image_dir):
        exit("Image directory '" + image_dir + "' not found.")

    images = []
    labels = []

    sub_dirs = [x[0] for x in os.walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))

            if (percentage_hash < validation_percentage) == validation:
                images.append(file_name)
                labels.append(label_name)

    class_count = len(set(labels))
    if class_count == 0:
        exit('No valid folders of images found at ' + image_dir)
    if class_count == 1:
        exit('Only one valid folder of images found at ' + image_dir +
             ' - multiple classes are needed for classification.')

    return images, labels


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, learning_rate):
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
                                        name='final_weights')
            variable_summaries(layer_weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.histogram_summary(layer_name + '/pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.histogram_summary(final_tensor_name + '/activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits, ground_truth_input)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.scalar_summary('cross entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', evaluation_step)
    return evaluation_step


def get_random_bottlenecks(images, labels, classes, how_many, sess, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []

    indices = random.sample(range(len(images)), how_many)
    for i in indices:
        image = images[i]
        label_index = classes.index(labels[i])
        jpeg_data = open(image, 'rb').read()

        bottleneck_values = sess.run(bottleneck_tensor,
                                     {jpeg_data_tensor: jpeg_data})
        bottleneck = np.squeeze(bottleneck_values)

        ground_truth = np.zeros(len(classes), dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def main(image_dir, model_dir, output_dir, steps, batch):
    # Setup the directory we'll write summaries to for TensorBoard
    if os.path.exists(SUMMARIES_DIR):
        shutil.rmtree(SUMMARIES_DIR)
    os.mkdir(SUMMARIES_DIR)

    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = \
        create_inception_graph(model_dir)

    images, labels = create_image_lists(image_dir, VALIDATION_PERCENTAGE, validation=False)
    classes = sorted(set(labels))

    class_count = len(classes)
    for k, g in groupby(sorted(zip(labels, images)), operator.itemgetter(0)):
        print('%s: %s' % (k, len(list(g))))

    sess = tf.Session()

    # Add the new layer that we'll be training
    (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) \
        = add_final_training_ops(class_count, FINAL_TENSOR_NAME, bottleneck_tensor, LEARNING_RATE)

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/train', sess.graph)
    validation_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(steps):
        print('%s: Step %d' % (datetime.now(), i))

        # Get a batch of input bottleneck values, either calculated fresh every time
        # with distortions applied, or from the cache stored on disk.
        train_bottlenecks, train_ground_truth \
            = get_random_bottlenecks(images, labels, classes, batch, sess, jpeg_data_tensor, bottleneck_tensor)

        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.
        train_summary, _ = sess.run([merged, train_step],
                                    feed_dict={bottleneck_input: train_bottlenecks,
                                               ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)

        if (i % EVAL_STEP_INTERVAL) == 0 or i + 1 == steps:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,

                                                       cross_entropy_value))

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_images, test_labels = create_image_lists(image_dir, VALIDATION_PERCENTAGE, validation=True)
    test_bottlenecks, test_ground_truth \
        = get_random_bottlenecks(test_images, test_labels, classes, len(test_images), sess, jpeg_data_tensor, bottleneck_tensor)
    test_accuracy = sess.run(
        evaluation_step,
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    # Write out the trained graph and labels with the weights stored as constants.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
    with open(os.path.join(output_dir, 'output_graph.pb'), 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with open(os.path.join(output_dir, 'output_labels.txt'), 'w') as f:
        f.write('\n'.join(classes) + '\n')
