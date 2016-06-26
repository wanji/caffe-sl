#!/usr/bin/env python
# coding: utf-8

"""
   File Name: tools/merge.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Sep 11 12:40:33 2015 CST
"""
DESCRIPTION = """
"""

import os
import argparse
import logging

import h5py
import numpy as np
from scipy.io import loadmat, savemat


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
    parser.add_argument('dir', type=str,
                        help='dir containing matfiles')
    parser.add_argument('out', type=str,
                        help='single output file')
    parser.add_argument("--out_type", type=str, default="mat",
                        help="output type: mat | h5")
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def load_feat(path):
    """
    Load feature from mat file or dir contains mat files.
    """
    if os.path.isdir(path):
        files = sorted(
            [fname for fname in os.listdir(path)
             if fname.startswith('feat_') and fname.endswith('.mat')],
            key=lambda x: int(x.split("_")[-1].split(".")[0]))
        v_feat = []
        for fname in files:
            fpath = os.path.join(path, fname)
            v_feat.append(loadmat(fpath)['feat'])
        return np.vstack(tuple(v_feat))
    else:
        return loadmat(path)['feat']


def main(args):
    """ Main entry.
    """
    logging.info("Loading `mat` files ...")
    feat = load_feat(args.dir)
    logging.info("\tDone!")

    logging.info("Saving `{}` files ...".format(args.out_type))
    if args.out_type == "mat":
        savemat(args.out, {'feat': feat})
    elif args.out_type == "h5":
        fp = h5py.File(args.out, 'w')
        fp.create_dataset('/feat', feat.shape, data=feat)
        fp.close()
    else:
        raise Exception("Unsupported output type: {}".format(args.out_type))

    logging.info("\tDone!")


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
