#!/usr/bin/env python
# coding: utf-8

"""
   File Name: extract_mnist.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Oct 13 21:13:51 2015 CST
"""
DESCRIPTION = """
"""

import os
import argparse
import logging

from struct import unpack
import numpy as np
from scipy.misc import imsave


def runcmd(cmd):
    """ Run command.
    """

    logging.info("%s" % cmd)
    os.system(cmd)


def getargs():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('mnist_dir', type=str,
                        nargs='?', default="./data/mnist",
                        help='directory of mnist dataset')
    parser.add_argument('out_dir', type=str,
                        nargs='?', default="./examples/mnist_sl/data",
                        help='output folder')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def mnist_load_image(path):
    with open(path, 'rb') as f:
        m, n, w, h = unpack('iiii', f.read(16)[::-1])[::-1]
        data = f.read()
    images = np.array(bytearray(data), np.uint8).reshape(n, w, h)
    return images


def mnist_load_label(path):
    with open(path, 'rb') as f:
        m, n = unpack('ii', f.read(8)[::-1])[::-1]
        data = f.read()
    images = np.array(bytearray(data), np.uint8).reshape(n)
    return images


def main(args):
    """ Main entry.
    """
    logging.info("Loading t10k ...")
    t10k_images = mnist_load_image(os.path.join(args.mnist_dir,
                                                't10k-images-idx3-ubyte'))
    t10k_labels = mnist_load_label(os.path.join(args.mnist_dir,
                                                't10k-labels-idx1-ubyte'))
    logging.info("Done!")
    logging.info("Loading train ...")
    train_images = mnist_load_image(os.path.join(args.mnist_dir,
                                                 'train-images-idx3-ubyte'))
    train_labels = mnist_load_label(os.path.join(args.mnist_dir,
                                                 'train-labels-idx1-ubyte'))
    logging.info("Done!")

    t10k_dir = os.path.join(args.out_dir, 't10k')
    train_dir = os.path.join(args.out_dir, 'train')
    t10k_lst = os.path.join(args.out_dir, 't10k.lst')
    train_lst = os.path.join(args.out_dir, 'train.lst')

    if not os.path.exists(t10k_dir):
        os.makedirs(t10k_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    logging.info("Saving t10k ...")
    with open(t10k_lst, 'w') as f:
        for i in xrange(t10k_images.shape[0]):
            impath = os.path.join(t10k_dir, '%06d.png' % i)
            imsave(impath, t10k_images[i])
            f.write('%s %d\n' % ('%06d.png' % i, t10k_labels[i]))
    logging.info("Done!")

    logging.info("Saving train ...")
    with open(train_lst, 'w') as f:
        for i in xrange(train_images.shape[0]):
            impath = os.path.join(train_dir, '%06d.png' % i)
            imsave(impath, train_images[i])
            f.write('%s %d\n' % ('%06d.png' % i, train_labels[i]))
    logging.info("Done!")


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
