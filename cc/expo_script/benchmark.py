# encoding=utf-8
import lmdb
from os.path import join
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
import sklearn
import hashlib
from collections import OrderedDict


parser = argparse.ArgumentParser(description='Calculate MAE/MSE')
parser.add_argument('--model', type=str, required=False, default=join(os.path.dirname('__file__'),'tmp', 'test.prototxt'))
parser.add_argument('--weights', type=str, required=False, default=join(os.path.dirname('__file__'), 'model_0_iter_10000.caffemodel'))
args = parser.parse_args()

# class args(object):
#     def __init__(self):
#         return
# args.model = join('/home/aachenhang/dataset/cc', 'tmp', 'test.prototxt')
# args.weights = join('/home/aachenhang/dataset/cc', 'model_1_iter_10000.caffemodel')



caffe.set_mode_gpu()
net = caffe.Net(args.model, args.weights, caffe.TEST)






test_simple_number = 599
test_batch_size = 1
Iter = test_simple_number / test_batch_size


#get the ground truth from the frame directory
def is_image(im):
    return ('.jpg' in im) or ('.JPG' in im) or ('.PNG' in im) or ('.png' in im)
test_list = [img for img in os.listdir('/home/aachenhang/dataset/frames_Expo/test/') if is_image(img)]
assert (len(test_list) != 0)

#Shuffle the test set
img_hash = {hashlib.md5(img).hexdigest(): img for img in test_list}
img_hash = OrderedDict(sorted(img_hash.items()))
test_list = img_hash.values()

label = np.array([])
for img in test_list:
    label_one_simple = int(re.sub("[^0-9]", "", img.split('n')[-1]))
    label = np.append(label, label_one_simple)


#get the prediction from the test network
pred = np.array([])
with open(join(os.path.dirname('__file__'),'..', 'data','ExpoDB', 'Ratio0.4','db.info')) as db_info:
    l = db_info.readlines()[1].split(',')  
    scale = float(re.sub("[^0-9\.]", "", l[-1]))


for i in range(Iter):
    out = net.forward()
    """
    pred和label的shape都是[20, 41, 1, 1]，每个batch有20个样本
    """
    pred_in_one_batch = net.blobs['pred'].data
    for j in range(test_batch_size):
        pred_one_simple = pred_in_one_batch[j,:,0,0].argsort()[-1]
        pred_one_simple = round(pred_one_simple * scale)
        pred = np.append(pred, pred_one_simple)



#save pred and label
np.savetxt('pred.txt', pred)
print 'save the pred at "pred.txt"'
np.savetxt('label.txt', label)
print 'save the label at "label.txt"'

#calculate MAE and MSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

print "MAE", mean_absolute_error(pred, label)
print "MSE", mean_squared_error(pred, label)
