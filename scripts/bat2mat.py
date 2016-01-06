#!/usr/bin/env python
# coding: utf-8

"""
   File Name: bat2mat.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Dec 28 20:25:49 2015 CST
"""
DESCRIPTION = """
Convert features extracted by extract_features_to_dir to `.mat` files.
"""

import os
import argparse
import logging
import numpy as np
from scipy.io import savemat


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
    parser.add_argument('mat_dir', type=str,
                        help='dir for saving mat files')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """
    os.makedirs(args.mat_dir)
    with open(os.path.join(args.bat_dir, 'meta'), 'r') as mf:
        dtype = np.dtype(mf.readline().strip())
        b, c, h, w = [int(x) for x in mf.readline().split()]
        first_idx = 0
        for line in mf:
            feat = np.fromfile(os.path.join(args.bat_dir, line.strip()), dtype)
            feat = feat.reshape(-1, c, h, w)
            savemat(os.path.join(args.mat_dir, "feat_%d.mat" % (first_idx)),
                    {'feat': feat})
            first_idx += feat.shape[0]


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
