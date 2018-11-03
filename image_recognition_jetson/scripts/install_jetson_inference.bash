#!/bin/bash

BUILD_ROOT=$PWD

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory:       $BUILD_ROOT"

function build_jetson-inference {
    # enter the jetson-inference directory
    cd ~/jetson-inference

    # create and enter a build directory
    mkdir -p build
    cd build

    # build and install the jetson-inference library
    cmake ../ -DCMAKE_BUILD_TYPE=Release
    make
    sudo make install
}

if [ ! -d ~/jetson-inference ]
then
    # clone the jetson-inference repository
    git clone https://github.com/dusty-nv/jetson-inference ~/jetson-inference
    build_jetson-inference
elif cd ~/jetson-inference &&
    git checkout master -q &&
    git fetch origin master -q &&
    [ `git rev-list HEAD...origin/master --count` != 0 ] &&
    git merge origin/master #If something needed to be updated
then
    echo "[Pre-build]  needs update"

    build_jetson-inference

    echo "[Pre-build]  has been updated!"
else
    # Show up-to-date message
    echo "[Pre-build]  up to date"
fi
