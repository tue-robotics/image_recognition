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
from argparse import ArgumentParser
from datetime import datetime
from itertools import groupby
from sortedcontainers import SortedSet

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.util import compat
from tensorflow_ros.retrain import defaults, main


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_dir', help='Image folder')

    parser.add_argument('model_dir', help='Folder where inception is stored')
    parser.add_argument('output_dir', help='Where to save the trained graph')

    parser.add_argument('--steps', type=int, default=defaults.steps)
    parser.add_argument('--batch', type=int, default=defaults.batch, help='How many images to train on at a time')

    args = parser.parse_args()

    main(**vars(args))
