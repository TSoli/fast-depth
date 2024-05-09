#!/usr/bin/env bash

source /opt/ros/humble/setup.sh
source ./install/local_setup.sh
source ./venv/bin/activate
ros2 run fast_depth fast_depth --ros-args -p ckpt:=fast_depth.pth.tar
