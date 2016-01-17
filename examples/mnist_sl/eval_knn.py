#!/usr/bin/env python
# coding: utf-8

"""
   File Name: eval_knn.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Jan  4 10:36:31 2016 CST
"""
DESCRIPTION = """
"""

import os
import argparse
import logging

import numpy as np
from scipy.io import loadmat

from bottleneck import argpartsort


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
    parser.add_argument('train_feat', type=str,
                        help='training features')
    parser.add_argument('test_feat', type=str,
                        help='test features')
    parser.add_argument('train_label', type=str,
                        help='training features')
    parser.add_argument('test_label', type=str,
                        help='test features')
    parser.add_argument("--Ks", type=int, nargs='+', default=[1],
                        help="Ks")
    parser.add_argument("--num_test", type=int,
                        help="number of test images")
    parser.add_argument("--dist_type", type=str, default='euclidean',
                        help="type of distance (euclidean, cosine, dotproduct)")
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def normalization(feat):
    """
    Feature normalization
    """
    feat = feat.reshape(feat.shape[0], -1)
    return feat / np.sqrt(np.sum(feat**2, axis=1)).reshape((feat.shape[0], 1))


def knn_classifer(knn_label):
    # if knn_label.shape[1] is 1:
    #     return knn_label[:, 0]

    num_label = knn_label.max() + 1
    num_test = knn_label.shape[0]
    pred = np.empty(num_test, np.int)
    for i in xrange(num_test):
        label_hist = np.histogram(knn_label[i], range(num_label+1))
        pred[i] = label_hist[0].argmax()
    return pred


def DotProduct(feat, query):
    """ dot product distance.
    """
    return -query.dot(feat.T)


def Euclidean(feat, query):
    """ Euclidean distance.
    """
    (nQ, D) = query.shape
    (N, D) = feat.shape
    dotprod = query.dot(feat.T)
    qryl2norm = (query ** 2).sum(1).reshape(-1, 1)
    featl2norm = (feat ** 2).sum(1).reshape(1, -1)

    return qryl2norm + featl2norm - 2 * dotprod


def main(args):
    """ Main entry.
    """

    logging.info("Loading features")
    # trn_feat = normalization(loadmat(args.train_feat)['feat'])
    # tst_feat = normalization(loadmat(args.test_feat)['feat'])
    trn_feat = loadmat(args.train_feat)['feat'].astype(np.float)
    tst_feat = loadmat(args.test_feat)['feat'].astype(np.float)
    logging.info("\tDone!")

    logging.info("Loading labels")
    with open(args.train_label, 'r') as lbf:
        trn_label = np.array([int(line.split()[1]) for line in lbf])
    with open(args.test_label, 'r') as lbf:
        tst_label = np.array([int(line.split()[1]) for line in lbf])
    logging.info("\tDone!")
    # print trn_feat.shape
    # print tst_feat.shape
    # print trn_label.shape
    # print tst_label.shape

    num_test = tst_feat.shape[0]
    if args.num_test is not None and args.num_test < tst_feat.shape[0]:
        num_test = args.num_test
        tst_feat = tst_feat[:num_test]
        tst_label = tst_label[:num_test]

    logging.info("Computing distances")
    if args.dist_type == "euclidean":
        dist = Euclidean(trn_feat, tst_feat)
    elif args.dist_type == "cosine":
        trn_feat = normalization(trn_feat)
        tst_feat = normalization(tst_feat)
        dist = DotProduct(trn_feat, tst_feat)
    elif args.dist_type == "dotproduct":
        dist = DotProduct(trn_feat, tst_feat)
    else:
        raise Exception("Invalid distance type.")
    logging.info("\tDone!")

    maxK = min(max(args.Ks), trn_feat.shape[0])

    logging.info("Sorting")
    idxs = np.empty((num_test, maxK), np.int)
    for i in xrange(num_test):
        cur_idxs = argpartsort(dist[i], maxK)[:maxK]
        idxs[i, :] = cur_idxs[dist[i][cur_idxs].argsort()]
    logging.info("\tDone!")

    logging.info("Labeling")
    knn_label = np.empty((num_test, maxK), np.int)
    for i in xrange(num_test):
        knn_label[i, :] = trn_label[idxs[i]]
    logging.info("\tDone!")

    logging.info("Evaluating")
    for k in args.Ks:
        pred = knn_classifer(knn_label[:, :k])
        accy = (pred == tst_label).sum() * 100.0 / tst_label.shape[0]
        logging.info("\t%4d-NN classifier: %.2f" % (k, accy))
    logging.info("\tDone")


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
