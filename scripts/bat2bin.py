#!/usr/bin/env python
# coding: utf-8

"""
   File Name: bat2bin.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Wed Dec 30 11:25:40 2015 CST
"""
DESCRIPTION = """
Convert features extracted by extract_features_to_dir to binary files.
"""

import os
import argparse
import logging
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
    parser.add_argument('bat_dir', type=str,
                        help='dir containing feature batches')
    parser.add_argument('bin_dir', type=str,
                        help='dir for saving binary files')
    parser.add_argument('img_lst', type=str,
                        help='image list')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """
    if not os.path.exists(args.bin_dir):
        os.makedirs(args.bin_dir)

    logging.info("Loading image list ...")
    with open(args.img_lst, 'r') as lstf:
        img_lst = [line.split()[0] for line in lstf]
    logging.info("\tDone!")

    idx = 0
    with open(os.path.join(args.bat_dir, 'meta'), 'r') as mf:
        dtype = np.dtype(mf.readline().strip())
        b, c, h, w = [int(x) for x in mf.readline().split()]
        dim = c * h * w
        logging.info("Converting ...")
        for line in mf:
            bat_path = os.path.join(args.bat_dir, line.strip())
            logging.info("\t{}".format(bat_path))
            if not os.path.exists(bat_path):
                idx += b
                continue

            feat = np.fromfile(bat_path, dtype)
            feat = feat.reshape(-1, dim)
            num = min(feat.shape[0], len(img_lst)-idx)
            for i in xrange(num):
                feat[i].tofile(os.path.join(args.bin_dir, img_lst[idx]))
                idx += 1
            os.unlink(bat_path)

        logging.info("\tDone!")

    with open(os.path.join(args.bin_dir, 'meta'), 'w') as mf:
        mf.write("{}\n".format(dtype))
        mf.write("{}\t{}\t{}\t{}\n".format(idx, c, h, w))
        mf.write("{}\n".format("\n".join(img_lst)))


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
