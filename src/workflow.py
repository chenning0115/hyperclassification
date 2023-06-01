import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer, BaseTrainer, CrossTransformerTrainer
import evaluation
from utils import check_convention, config_path_prefix

DEFAULT_RES_SAVE_PATH_PREFIX = "./res/"

def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)
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
    print('99999')

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(trainX, trainY)
    eval_res = trainer.final_eval(testX, testY)
    pred_all = trainer.test(allX)
    pred_matrix = dataloader.reconstruct_pred(pred_all)
    print('aaaaa')

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)
    print('bbbbb')

    return recorder 




include_path = [
    # "indian_ssftt.json",
    # 'indian_casst.json'
    # 'indian_cross_param.json',
    # 'indian_cross_param_use.json',

    # 'pavia_cross_param_use.json',
    # 'pavia_cross_param.json',

    # 'salinas_cross_param_use.json'
    # 'pavia_cross_param_use.json',
    
    # 'pavia_diffusion.json',
    # 'salinas_diffusion.json',

    # 'indian_ssftt.json',

    # for batch process 
    'temp.json'
]

def run_all():
    print('11111')
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    print('22222')
    for name in include_path:
        convention = check_convention(name)
        path_param = '%s/%s' % (config_path_prefix, name)
        print('33333')
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        print('44444')
        uniq_name = param.get('uniq_name', name)
        print('start to train %s...' % uniq_name)
        if convention:
            train_convention_by_param(param)
        else:
            print('55555')
            train_by_param(param)
            print('66666')
        print('model eval done of %s...' % uniq_name)
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        print('77777')
        recorder.to_file(path)
        print('88888')


if __name__ == "__main__":
    run_all()
    
    




