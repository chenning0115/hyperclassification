from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import tifffile as tiff
import os, sys


def load_data(data_sign, data_path_prefix):
    if data_sign == "Indian":
        data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    elif data_sign == "Pavia":
        data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt'] 
    elif data_sign == "Houston":
        data = sio.loadmat('%s/Houston.mat' % data_path_prefix)['img']
        labels = sio.loadmat('%s/Houston_gt.mat' % data_path_prefix)['Houston_gt']
    elif data_sign == 'Salinas':
        data = sio.loadmat('%s/Salinas_corrected.mat' % data_path_prefix)['salinas_corrected']
        labels = sio.loadmat('%s/Salinas_gt.mat' % data_path_prefix)['salinas_gt']
    return data, labels

def gen(data_sign, train_num_per_class, data_path_prefix, max_percent=0.5):
    data, labels = load_data(data_sign, data_path_prefix)
    h, w, c = data.shape
    class_num = labels.max()
    class2data = {}
    for i in range(h):
        for j in range(w):
            if labels[i,j] > 0:
                if labels[i, j] in class2data:
                    class2data[labels[i,j]].append([i, j])
                else:
                    class2data[labels[i,j]] = [[i,j]]

    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)
    for cl in range(class_num):
        class_index = cl + 1
        ll = class2data[class_index]
        all_index = list(range(len(ll)))
        real_train_num = train_num_per_class
        if len(all_index) <= train_num_per_class:
            real_train_num = int(len(all_index) * max_percent) 
        select_train_index = set(random.sample(all_index, real_train_num))
        for index in select_train_index:
            item = ll[index]
            TR[item[0], item[1]] = class_index
    TE = labels - TR
    target = {}
    target['TE'] = TE
    target['TR'] = TR
    target['input'] = data
    return target


def run():
    signs = ['Indian', 'Pavia', 'Salinas']
    #signs = ['Salinas']
    # signs = ['Indian']
    data_path_prefix = '../../data'
    train_num_per_class_list = [10, 20, 30, 40, 50, 60, 70, 80]
    for data_sign in signs:
        for train_num_per_class in train_num_per_class_list:
            save_path = '../../data/%s/%s_%s_split.mat' %(data_sign, data_sign, train_num_per_class)
            target = gen(data_sign, train_num_per_class, data_path_prefix)
            sio.savemat(save_path, target)
            print('save %s done.' % save_path)

def run_miniGCN():
    signs = ['Pavia']
    data_path_prefix = '../../data'
    train_num_per_class_list = [10]
    for data_sign in signs:
        for train_num_per_class in train_num_per_class_list:
            save_path_prefix = '../../data/miniGCN/%s/%s' %(data_sign, train_num_per_class)
            if not os.path.exists(save_path_prefix):
                os.makedirs(save_path_prefix)
            target = gen(data_sign, train_num_per_class, data_path_prefix)


            data, TR, TE = target['input'], target['TR'], target['TE']
            data_new = data.transpose((2,0,1))
            TALL = TE + TR
            TALL[TALL == 0] = 1
            print(data_new.shape, TR.shape, TE.shape, TALL.shape)
            tiff.imsave('%s/data.tiff' % save_path_prefix, data_new)
            tiff.imsave('%s/TR.tiff' % save_path_prefix, TR)
            tiff.imsave('%s/TE.tiff' % save_path_prefix, TE)
            tiff.imsave('%s/ALL.tiff' % save_path_prefix, TALL)

if __name__ == "__main__":
    # run()
    run_miniGCN()

