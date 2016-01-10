#!/usr/bin/env sh
# This script extracts the images from mnist dataset,
# and generate triplets for similarity learning.

EXAMPLE=examples/mnist_sl
DATA_ORI=data/mnist
DATA_NEW=$EXAMPLE/data


rm -fr $DATA_NEW

$EXAMPLE/extract_mnist.py $DATA_ORI $DATA_NEW

echo "Generate validation triplets ..."
$EXAMPLE/generate_triplet.py $DATA_NEW/t10k.lst $DATA_NEW/t10k.tri 100000
echo "Generate training triplets ..."
$EXAMPLE/generate_triplet.py $DATA_NEW/train.lst $DATA_NEW/train.tri 1000000

echo "Generate validation batches ..."
$EXAMPLE/generate_batch.py $DATA_NEW/t10k.lst $DATA_NEW/t10k.bat 10 10
echo "Generate training batches ..."
$EXAMPLE/generate_batch.py $DATA_NEW/train.lst $DATA_NEW/train.bat 10 10

echo "Done!"
