#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['skybiometry_ros'],
    package_dir={'': 'src'},
    scripts=['scripts/get_face_properties.py', 'scripts/face_properties_node.py'],
)

setup(**d)
