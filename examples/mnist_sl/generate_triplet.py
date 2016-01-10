#!/usr/bin/env python
# coding: utf-8

"""
   File Name: gen_lst.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Sun Oct 11 18:05:16 2015 CST
"""
DESCRIPTION = """
"""

import os
import argparse
import logging
import random

import numpy as np


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
    parser.add_argument('ori_lst', type=str,
                        help='original image list')
    parser.add_argument('tri_lst', type=str,
                        help='triplet list')
    parser.add_argument('num_tri', type=int, nargs='?', default=1000000,
                        help='num of triplets')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """
    logging.info("Loading image lists ...")
    v_info = []
    img_lst = dict()
    with open(args.ori_lst, 'r') as orif:
        v_info = [line.split() for line in orif]
    for impath, label in v_info:
        if label not in img_lst:
            img_lst[label] = []
        img_lst[label].append(impath)
    logging.info("Done!")

    num_img = len(v_info)
    v_imgid = range(num_img)
    v_negid = range(num_img)
    qry_idx = num_img
    neg_idx = -1

    logging.info("Generating triplets ...")
    with open(args.tri_lst, 'w') as trif:
        for i in xrange(args.num_tri):
            if i % 10000 == 0:
                logging.info("\tfinished %6.2f%% of %d" % (
                    100.0 * i / args.num_tri, args.num_tri))

            if qry_idx == num_img:
                random.shuffle(v_imgid)
                qry_idx = 0

            qry_imgid = v_imgid[qry_idx]
            qry, label = v_info[qry_imgid]
            for j in xrange(10):
                pos = random.choice(img_lst[label])
                if pos != label:
                    break

            while True:
                neg_idx += 1
                if neg_idx == num_img:
                    random.shuffle(v_negid)
                    neg_idx = 0
                neg_imgid = v_negid[neg_idx]
                neg, neg_label = v_info[neg_imgid]
                if neg_label != label:
                    break

            trif.write("%s\t%s\t%s\n" % (qry, pos, neg))

            qry_idx += 1
    logging.info("\tfinished 100.00%% of %d" % (args.num_tri))
    logging.info("Done!")


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
