#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/mnist_sl/lenet_batch_solver.prototxt \
  -gpu 0
