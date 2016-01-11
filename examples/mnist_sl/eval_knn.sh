#!/bin/bash

#########################################################################
# File Name: eval_knn.sh
# Author: Wan Ji
# mail: wanji@live.com
# Created Time: Sun 10 Jan 2016 05:47:37 PM CST
#########################################################################

BASEDIR=$(dirname $0)

extract() {
  MD=$1
  PROTOTXT=$2
  FEAT_NAME=$3
  FEAT_DIR=$4

  mkdir -p $FEAT_DIR

  rm -fr $FEAT_DIR/$FEAT_NAME
  ./build/bin/extract_features_to_dir $MD $PROTOTXT \
    $FEAT_NAME $FEAT_DIR 100 GPU 0
  rm -fr $FEAT_DIR/$FEAT_NAME"_mat"
  ./scripts/bat2mat.py $FEAT_DIR/$FEAT_NAME       $FEAT_DIR/$FEAT_NAME"_mat"
  rm -fr $FEAT_DIR/$FEAT_NAME".mat"
  ./scripts/dir2mat.py $FEAT_DIR/$FEAT_NAME"_mat" $FEAT_DIR/$FEAT_NAME".mat"
}


FEAT_NAME=ip2

for MD_NAME in batch_iter_6000 naive_iter_2000 batch_iter_2000; do
# for MD_NAME in naive_iter_2000 batch_iter_2000; do
  for sp in train test; do
    MD=examples/mnist_sl/lenet_$MD_NAME.caffemodel
    PROTOTXT=examples/mnist_sl/lenet_$sp.prototxt
    extract $MD $PROTOTXT $FEAT_NAME examples/mnist_sl/feature/$MD_NAME/$sp
  done
  ./examples/mnist_sl/eval_knn.py \
    examples/mnist_sl/feature/$MD_NAME/train/$FEAT_NAME.mat \
    examples/mnist_sl/feature/$MD_NAME/test/$FEAT_NAME.mat \
    examples/mnist_sl/data/train.lst \
    examples/mnist_sl/data/t10k.lst \
    --Ks `seq 1 10` `seq 20 10 100` `seq 200 100 1000`
done
