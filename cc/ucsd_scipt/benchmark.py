
# coding: utf-8

# In[1]:


import sys, os, re, urllib
#sys.path.insert(0, '/home/zt/LDLF/caffe-ldl/python')
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
import sklearn

# In[2]:


parser = argparse.ArgumentParser(description='Calculate MAE/MSE')
parser.add_argument('--model', type=str, required=False, default=join('/home/aachenhang/dataset/cc', 'tmp', 'test.prototxt'))
parser.add_argument('--weights', type=str, required=False, default=join('/home/aachenhang/dataset/cc', 'model_1_iter_10000.caffemodel'))
args = parser.parse_args()

# class args(object):
#     def __init__(self):
#         return
# args.model = join('/home/aachenhang/dataset/cc', 'tmp', 'test.prototxt')
# args.weights = join('/home/aachenhang/dataset/cc', 'model_1_iter_10000.caffemodel')


# In[3]:

caffe.set_mode_gpu()
net = caffe.Net(args.model, args.weights, caffe.TEST)


# In[4]:


test_simple_number = 1200
test_batch_size = 20
Iter = test_simple_number / test_batch_size

pred = np.array([])
label = np.array([])


for i in range(Iter):
    out = net.forward()
    """
    pred和label的shape都是[20, 41, 1, 1]，每个batch有20个样本
    """
    pred_in_one_batch = net.blobs['pred'].data
    label_in_one_batch = net.blobs['label'].data
    for j in range(test_batch_size):
        pred_one_simple = pred_in_one_batch[j,:,0,0].argsort()[-1]
        label_one_simple = label_in_one_batch[j,:,0,0].argsort()[-1]
        pred = np.append(pred, pred_one_simple)
        label = np.append(label, label_one_simple)




np.savetxt('pred.txt', pred)
print 'save the pred at "pred.txt"'
np.savetxt('label.txt', label)
print 'save the label at "label.txt"'

from sklearn.metrics import mean_absolute_error, mean_squared_error

print "MAE", mean_absolute_error(pred, label)
print "MSE", mean_squared_error(pred, label)