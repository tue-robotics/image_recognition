#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['openface_ros'],
    package_dir={'': 'src'},
    scripts=['scripts/face_recognition_node.py'],
)

setup(**d)
