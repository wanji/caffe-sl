#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist_sl/lenet_naive_solver.prototxt -gpu 1
