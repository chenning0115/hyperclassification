from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import tifffile as tiff


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

def gen(data_sign, train_num_per_class, max_percent=0.5):
    data, labels = load_data(data_sign)
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
    signs = ['Indian', 'Pavia', 'Houston', 'Salinas']
    data_path_prefix = '../../data'
    train_num_per_class_list = [10, 20, 30, 40, 50, 60, 70, 80]
    for data_sign in signs:
        for train_num_per_class in train_num_per_class_list:
            save_path = '../../data/%s/%s_80_split.mat' %(data_sign, data_sign)
            target = gen(data_sign, train_num_per_class)
            sio.savemat(save_path, target)
            print('save %s done.' % save_path)



if __name__ == "__main__":
    run()

