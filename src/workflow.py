import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer, BaseTrainer, CrossTransformerTrainer
import evaluation

DEFAULT_RES_SAVE_PATH_PREFIX = "./res/"

def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader, unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader, test_loader)
    eval_res = trainer.final_eval(test_loader)
    # pred_all, y_all = trainer.test(all_loader)
    # pred_matrix = dataloader.reconstruct_pred(pred_all)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    # recorder.record_pred(pred_matrix)

    return recorder

def train_convention_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    trainX, trainY, testX, testY, allX = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(trainX, trainY)
    eval_res = trainer.final_eval(testX, testY)
    pred_all = trainer.test(allX)
    pred_matrix = dataloader.reconstruct_pred(pred_all)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)

    return recorder 




include_path = [
    'pavia_contra_mask.json',
]

def check_convention(name):
    for a in ['knn', 'random_forest', 'svm']:
        if a in name:
            return True
    return False

def run_all():
    save_path_prefix = './res/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        convention = check_convention(name)
        path_param = './params/%s' % name
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        print('start to train %s...' % name)
        if convention:
            train_convention_by_param(param)
        else:
            train_by_param(param)
        print('model eval done of %s...' % name)
        path = '%s/%s' % (save_path_prefix, name) 
        recorder.to_file(path)


if __name__ == "__main__":
    run_all()
    
    




