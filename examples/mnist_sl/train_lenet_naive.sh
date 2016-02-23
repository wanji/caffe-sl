#!/usr/bin/env sh

DATE=`date "+%y.%m.%d_%H.%M.%S"`

./build/tools/caffe train \
  --solver=examples/mnist_sl/lenet_naive_solver.prototxt \
  -gpu 0 \
  2>&1 | tee examples/mnist_sl/lenet_naive_solver.$DATE.log
