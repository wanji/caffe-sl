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
import math
import ipdb


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
    parser.add_argument('bat_lst', type=str,
                        help='batch list')
    parser.add_argument('group_size', type=int, nargs='?', default=5,
                        help='size of small image groups')
    parser.add_argument('repeat', type=int, nargs='?', default=1,
                        help='number of repeats')
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
    logging.info("\tDone!")

    v_groups = []

    group_size = args.group_size
    num_repeat = args.repeat

    logging.info("Generating groups ...")
    for label, imgs in img_lst.iteritems():
        logging.info("\t%s" % label)
        num_img = len(imgs)
        num_grp = int(math.ceil(num_img * num_repeat * 1.0 / group_size))
        if num_img < group_size:
            logging.warn("number of images with label " + label +
                         " is less than group size")
        idx = num_img
        for i in xrange(num_grp):
            if idx + group_size <= num_img:
                sel_img = imgs[idx:idx+group_size]
                idx += group_size
            else:
                sel_img = imgs[idx:]
                random.shuffle(imgs)
                idx = (idx + group_size) % num_img
                sel_img += imgs[:idx]

            v_groups.append("\n".join(["%s\t%s" % (img, label)
                                       for img in sel_img]))
    logging.info("\tDone!")

    logging.info("Shuffling groups ...")
    random.shuffle(v_groups)
    logging.info("\tDone!")

    logging.info("Saving groups ...")
    with open(args.bat_lst, 'w') as batf:
        batf.write("\n".join(v_groups))
        batf.write("\n")
    logging.info("\tDone!")


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
