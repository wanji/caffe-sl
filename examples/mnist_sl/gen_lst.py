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
                        help='IP address of server')
    parser.add_argument('tri_lst', type=str,
                        help='IP address of server')
    parser.add_argument('num_tri', type=int, nargs='?', default=1000000,
                        help='num of triplets')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """
    img_lst = dict()
    with open(args.ori_lst, 'r') as orif:
        for line in orif:
            impath, imid = line.split()
            if imid not in img_lst:
                img_lst[imid] = []
            img_lst[imid].append(impath)

    with open(args.tri_lst, 'w') as trif:
        for i in xrange(args.num_tri):
            if i % 10000 == 0:
                logging.info("\tfinished %6.2f%% of %d" % (
                    100.0 * i / args.num_tri, args.num_tri))
            c1, c2 = np.random.choice(img_lst.keys(), 2)
            qry, pos = np.random.choice(img_lst[c1], 2)
            neg = np.random.choice(img_lst[c2])
            trif.write("%s\t%s\t%s\n" % (qry, pos, neg))
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
