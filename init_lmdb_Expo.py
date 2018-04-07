#!/usr/bin/python
# coding=utf-8

import sys, argparse, scipy, lmdb, shutil, hashlib
from PIL import Image
from collections import OrderedDict

import caffe
import numpy as np
from random import shuffle
import scipy
import os, re
from os.path import join, splitext, split, abspath, isdir
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser(description='Convert Expo database to LMDB')
parser.add_argument('--data', type=str, help='Expo database directory', required=False, default='frames_Expo/')
parser.add_argument('--ratio', type=float, help='Training set ratio', required=False, default=0.4)
parser.add_argument('--imsize', type=int, help='Image size', required=False, default=256)
parser.add_argument('--std', type=float, help='gaussian std', required=False, default=10)
parser.add_argument('--debug', type=bool, help='debug', required=False, default=False)
args = parser.parse_args()
if args.debug:
    import matplotlib.pyplot as plt
NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'

# 当前文件所在路
path = os.path.dirname(os.path.realpath(__file__))
def is_image(im):
    return ('.jpg' in im) or ('.JPG' in im) or ('.PNG' in im) or ('.png' in im)


max_count = max([int(re.sub("[^0-9]", "", img.split('n')[-1])) for img in os.listdir(join(args.data, 'test'))+os.listdir(join(args.data, 'train_frame')) if is_image(img)])
min_count = min([int(re.sub("[^0-9]", "", img.split('n')[-1])) for img in os.listdir(join(args.data, 'test'))+os.listdir(join(args.data, 'train_frame')) if is_image(img)])
mean_count = np.mean(np.array([int(re.sub("[^0-9]", "", img.split('n')[-1])) for img in os.listdir(join(args.data, 'test'))+os.listdir(join(args.data, 'train_frame')) if is_image('.JPG')], dtype=np.float))
#如果人数范围超过了100，则
scale = float(1.0*max_count / 100.0)

if(scale > 1):
    max_count = int(round(max_count / scale))
    min_count = int(round(min_count / scale))
    mean_count = int(round(mean_count / scale))


# 初始标签的分布为高斯分布，通过命令行参数可改变
gaussian = scipy.signal.gaussian(max_count - min_count + 1, args.std)


# 制作标签
def make_label(label_value):
    label_value = int(round(1.0*label_value / scale))
    label_value = label_value - min_count
    label_distr = np.zeros([max_count - min_count +1])
    mid =  np.ceil((max_count - min_count +1)/2)
    shift = int(label_value -mid)
    print 'shift' , shift
    if shift >0:
        label_distr[shift:] = gaussian[0:-shift]
    elif shift ==0:
        label_distr = gaussian
    else:
        label_distr[:shift] = gaussian[-shift:]
    label_distr = label_distr / np.sum(label_distr)
    print label_distr
    return label_distr

# 画出初始标签分布图
def draw_init_distr(min_count, max_cout):
    for num in range(min_count, max_count):
        img = str(num) + '.png'
        plt.plot(make_label(num))
        plt.savefig(os.path.join(path, 'init_distr', img))
        plt.close()

# 生成LMDB
def make_lmdb(db_path, img_list, data_type = 'image', phase = 'train'):
    if os.path.exists(db_path):
        # 如果已经存在该路径，则删去重建
        shutil.rmtree(db_path)
    os.makedirs(db_path)
    db = lmdb.open(db_path, map_size=int(1e12))
    with db.begin(write=True) as in_txn:
        for idx, img in enumerate(img_list):  # index , img
            if data_type == 'image':
                if(phase == 'train'):
                    data = np.array(Image.open(os.path.join(args.data, 'train_frame', img)), dtype=np.float)
                else:
                    data = np.array(Image.open(os.path.join(args.data, 'test', img)), dtype=np.float)
                data = scipy.misc.imresize(data, (360,288))

                data = data[:, :, ::-1]  # rgb to bgr
                data = data.transpose([2, 0, 1])
            elif data_type == 'count':
                count = int(re.sub("[^0-9]", "", img.split('n')[-1]))
                data = make_label(count).reshape([max_count - min_count + 1, 1, 1]).astype(np.float)
            data = caffe.io.array_to_datum(data)
            in_txn.put(IDX_FMT.format(idx), data.SerializeToString())
            if (idx + 1) % 10 == 0:
                print "Serializing to %s, %d of %d" % (
                    db_path, idx + 1, len(img_list))
    db.close()

if __name__ == '__main__':
    #img_list = [img for img in os.listdir(args.data) if is_image(img)]
    #assert (len(img_list) != 0)
    # sort img_list according to md5
    #img_hash = {hashlib.md5(img).hexdigest(): img for img in img_list}
    #img_hash = OrderedDict(sorted(img_hash.items()))
    #img_list = img_hash.values()
    
    #sort img_list according to name    
    #img_list = sorted(img_list)

    train_list = [img for img in os.listdir(join(args.data, 'train_frame')) if is_image(img)]
    assert (len(train_list) != 0)

    test_list = [img for img in os.listdir(join(args.data, 'test')) if is_image(img)]
    assert (len(test_list) != 0)

    base_dir = abspath(join(args.data, '..', 'cc', 'data', 'ExpoDB'))



    #Shuffle the train set
    img_hash = {hashlib.md5(img).hexdigest(): img for img in train_list}
    img_hash = OrderedDict(sorted(img_hash.items()))
    train_list = img_hash.values()
    print train_list


    # convert training data
    db_path = abspath(join(base_dir, 'Ratio' + str(args.ratio), 'train-img'))
    make_lmdb(db_path, train_list, 'image')
    db_path = join(base_dir, 'Ratio' + str(args.ratio), 'train-count') # path to training set
    make_lmdb(db_path, train_list, 'count')


    #Shuffle the test set
    img_hash = {hashlib.md5(img).hexdigest(): img for img in test_list}
    img_hash = OrderedDict(sorted(img_hash.items()))
    test_list = img_hash.values()
    

    # converting testing data
    db_path = abspath(join(base_dir, 'Ratio' + str(args.ratio), 'test-img'))
    make_lmdb(db_path, test_list, 'image', 'test')
    db_path = abspath(join(base_dir, 'Ratio' + str(args.ratio), 'test-count'))
    make_lmdb(db_path, test_list, 'count', 'test')

    with open(abspath(join(base_dir, 'Ratio' + str(args.ratio), 'db.info')), 'w') as db_info:
        db_info.write("Expo dataset LMDB info: TrainSet ratio=%f \n" % (args.ratio))
        db_info.write("nTrain=%d, nTest=%d, minCount=%d, maxCount=%d, meanCount=%f, imsize=%d, std=%f, scale=%f" \
                      % (800, 1200, min_count, max_count, mean_count, args.imsize, args.std, scale))
        if not isdir("data"):
            os.makedirs("data")
        if not isdir("data/ExpoDB"):
            os.symlink(base_dir, "data/ExpoDB")
            print "Make data symbol link at 'data/ExpoDB'."
