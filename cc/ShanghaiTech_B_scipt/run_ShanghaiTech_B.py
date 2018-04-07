#!/usr/bin/python
# coding=utf-8


import sys, os, re, urllib
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop
from caffe.proto import caffe_pb2
from os.path import join, splitext, abspath, exists, dirname, isdir, isfile
from datetime import datetime
from scipy.io import savemat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='CrowdNet')
parser.add_argument('--ratio', type=float, required=False, default=0.4)
parser.add_argument('--gpu', type=int, required=False, default=0)
parser.add_argument('--data', type=str, required=False, default='ShanghaiTech_B')
parser.add_argument('--tree', type=int, required=False, default=5)
parser.add_argument('--depth', type=int, required=False, default=7)
parser.add_argument('--drop', type=bool, required=False, default=False)
parser.add_argument('--nout', type=int, required=False, default=64)
args = parser.parse_args()

data_source = join(dirname(__file__), '..', 'data', 'ShanghaiTech_BDB', 'Ratio0.56')
with open(join(data_source, 'db.info')) as db_info:
  l = db_info.readlines()[1].split(',')
  nTrain = int(re.sub("[^0-9]", "", l[0]))
  nTest = int(re.sub("[^0-9]", "", l[1]))
  minCount = int(re.sub("[^0-9]", "", l[2]))
  maxCount = int(re.sub("[^0-9]", "", l[3]))

# params
test_batch_size = 20
ntree = args.tree
treeDepth = args.depth
maxIter = 10000
test_interval = 50
test_iter = int(np.ceil(nTest / test_batch_size))  # 测试一个epoch需要的iter数

tmp_dir = join(dirname(__file__), 'tmp')


if __name__ == '__main__':
    caffe.set_mode_gpu()
    iter = 0
    mae = []
    solver = caffe.SGDSolver(join(tmp_dir, 'solver.prototxt'))
    # base_weights = join(dirname(__file__), 'bvlc_reference_caffenet.caffemodel')
    # solver.net.copy_from(base_weights)
    print "*********************************************"
    print "Summarize of net parameters:"
    for p in solver.net.params:
        param = solver.net.params[p][0].data[...]
        print "  layer \"%s\":, parameter[0] mean=%f, std=%f" % (p, param.mean(), param.std())
    #raw_input("Press Enter to continue...")
    #solver.step(1)
    while iter < maxIter:  # 每次迭代
        solver.step(test_interval)  # 运行一个测试间隔
        solver.test_nets[0].share_with(solver.net)
        mae1 = np.float32(0.0)
        print "*********************************************"
        print mae
        print "*********************************************"
    #for layer_name, blob in solver.test_nets[0].blobs.iteritems():
        print solver.test_nets[0].blobs['pred'].data.shape
        for t in range(test_iter):  # 一个test_iter即为执行一个epoch的迭代次数
            ae = 0
            for j in range(solver.test_nets[0].blobs['pred'].data.shape[0]):
                a = solver.test_nets[0].blobs['pred'].data[j,:, 0, 0]
                b = solver.test_nets[0].blobs['label'].data[j,:, 0, 0]
                print a
                print b
                #raw_input('input the enter')
            print "pred"+str(j), np.where(a==max(a))[0][0]
            print "label"+str(j), np.where(b==max(b))[0][0]
#             ae += abs()
            ##mae1 += solver.test_nets[0].forward()['MAE']
    #raw_input("Press Enter to continue...")
        mae1 /= test_iter
        mae.append(mae1)
        iter = iter + test_interval
        print "Iter%d, currentMAE=%.4f, bestMAE=%.4f" % (iter, mae[-1], min(mae))
    mae = np.array(mae, dtype=np.float32)
    sav_fn = join(tmp_dir, "MAE-%s-Ratio%.1ftree%ddepth%dtime%s"%(
        args.data, args.ratio, ntree, treeDepth, datetime.now().strftime("M%mD%d-H%HM%MS%S")))
    np.save(sav_fn + '.npy', mae)
    mat_dict = dict({'mae': mae})
    # mat_dict.update(vars(args))  # save args to .mat
    savemat(sav_fn + '.mat', mat_dict)
    plt.plot(np.array(range(0, maxIter, test_interval)), mae)
    plt.title("ShanghaiTech_B: MAE vs Iter(best=%.4f)" % (mae.min()))
    plt.savefig(sav_fn + '.eps')
    plt.savefig(sav_fn + '.png')
    plt.savefig(sav_fn + '.svg')
    print args
    print "Best MAE=%.4f." % mae.min()
    print "Done! Results saved at \'" + sav_fn + "\'"
