import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import cross_transformer 
from models import conv1d
from models import conv2d
from models import conv3d
from models import contra_conv
import utils
from utils import recorder
from evaluation import HSIEvaluation
from augment import do_augment

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import itertools


class SKlearnTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.evalator = HSIEvaluation(param=params)


        self.model = None
        self.real_init()

    def real_init(self):
        pass
        

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY)
        print(self.model, "trian done.") 


    def final_eval(self, testX, testY):
        predictY = self.model.predict(testX)
        temp_res = self.evalator.eval(testY, predictY)
        print(temp_res['oa'], temp_res['aa'], temp_res['kappa'])
        return temp_res

    def test(self, testX):
        return self.model.predict(testX)

            
class SVMTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super(SVMTrainer, self).__init__(params)

    def real_init(self):
        kernel = self.net_params.get('kernel', 'rbf')
        gamma = self.net_params.get('gamma', 'scale')
        c = self.net_params.get('c', 1)
        self.model = svm.SVC(C=c, kernel=kernel, gamma=gamma)

class RandomForestTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n_estimators = self.net_params.get('n_estimators', 200)
        self.model = RandomForestClassifier(n_estimators = n_estimators, max_features="auto", criterion="entropy")

class KNNTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n = self.net_params.get('n', 10)
        self.model = KNeighborsClassifier(n_neighbors=n)



class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.evalator = HSIEvaluation(param=params)
        self.aug=params.get("aug",None)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 50
        self.real_init()

        self.cur_epoch = 0
        self.temp_unlabel_loader = None
    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)

    def next_unalbel_data(self): 
        index, (data, target) = next(self.temp_unlabel_loader)
        print(index)
        target = torch.ones_like(target) * -1
        return data.to(self.device), target.to(self.device)
        
    def train(self, train_loader, unlabel_loader, test_loader=None):
        self.temp_unlabel_loader = enumerate(itertools.cycle(unlabel_loader))
        epochs = self.params['train'].get('epochs', 100)
        use_unlabel = self.params['train'].get('use_unlabel', False)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            self.cur_epoch = epoch
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if use_unlabel:
                    unlabel_data, unlabel_target = self.next_unalbel_data()
                    # print(data.shape, unlabel_data.shape, target.shape, unlabel_target.shape)
                    data = torch.concatenate([data, unlabel_data], dim=0)
                    target = torch.concatenate([target, unlabel_target], dim=0)
                if self.aug:
                    left_data, right_data = do_augment(self.aug, data)
                    left_data, right_data = [d.to(self.device) for d in [left_data, right_data]]
                    outputs = self.net(data, left_data, right_data)
                    # print(outputs[1], outputs[2])
                else:
                    outputs = self.net(data, None, None)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
            
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        y_pred_test, y_test = self.test(test_loader)
        temp_res = self.evalator.eval(y_test, y_pred_test)
        return temp_res


    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.get_logits(self.net(inputs))
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test


class CrossTransformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(CrossTransformerTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = cross_transformer.HSINet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class ContraCrossTransformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(ContraCrossTransformerTrainer,self).__init__(params)

    def real_init(self):
        # net
        self.net = cross_transformer.HSINet(self.params).to(self.device)
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def infoNCE_diag(self, A_vecs, B_vecs, temperature=1, targets=None):
        '''
        targets: [batch]  dtype is int
        '''
        # print(A_vecs, B_vecs)
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        # tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        matrix_softmax = torch.softmax(matrix_logits, dim=1) # softmax by dim=1

        if np.random.randint(0,100) < 10 and targets is not None:
            tempb = matrix_softmax.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            # tempb = list(np.diag(tempb))
            # targets = list(np.diag(targets))
            tempb = list(tempb[0])
            targets = list(targets)
            tt = zip(tempb, targets)
            print(list(tt))
        matrix_log = -1 * torch.log(matrix_softmax)
        # here just use dig part
        loss_nce = torch.mean(torch.diag(matrix_log))
        return loss_nce

    def infoNCE(self, A_vecs, B_vecs, targets, temperature=15):
        '''
        targets: [batch]  dtype is int
        '''
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        # tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        matrix_softmax = torch.softmax(matrix_logits, dim=1) # softmax by dim=1
        matrix_log = -1 * torch.log(matrix_softmax)

        if np.random.randint(0,100) < 10 and targets is not None:
            tempb = matrix_softmax.detach().cpu().numpy()
            ltargets = targets.detach().cpu().numpy()
            # tempb = list(np.diag(tempb))
            # targets = list(np.diag(targets))
            tempb = list(tempb[0])
            ltargets = list(ltargets)
            tt = zip(tempb, ltargets)
            print(list(tt))
        l = targets.shape[0]
        tb = torch.repeat_interleave(targets.reshape([-1,1]), l, dim=1)
        tc = torch.repeat_interleave(targets.reshape([1,-1]), l, dim=0)
        mask_matrix = tb.eq(tc).int()
        # here just use dig part
        loss_nce = torch.sum(matrix_log * mask_matrix) / torch.sum(mask_matrix)
        return loss_nce



    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits, A_vecs, B_vecs = outputs
         
        weight_nce = 0.1
        if self.cur_epoch > 300:
            weight_nce = 0
        # loss_nce_1 = self.infoNCE_diag(A_vecs, B_vecs, temperature=10, targets=target) * weight_nce
        loss_nce_1 = self.infoNCE(A_vecs, B_vecs, target) * weight_nce
        loss_nce = loss_nce_1

        loss_main = nn.CrossEntropyLoss(ignore_index=-1)(logits, target) * (1 - weight_nce)

        print('nce=%s, main=%s, loss=%s' % (loss_nce.detach().cpu().numpy(), loss_main.detach().cpu().numpy(), (loss_nce + loss_main).detach().cpu().numpy()))

        return loss_nce + loss_main   

class Conv1dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv1d.Conv1d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class Conv2dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv2d.Conv2d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class Conv3dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv3d.Conv3d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class ContraConv3dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)
    
    def real_init(self):
        self.net=contra_conv.Conv3d(self.params).to(self.device)
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def infoNCE_diag(self, A_vecs, B_vecs, temperature=10):
        '''
        targets: [batch]  dtype is int
        '''
        # print(A_vecs, B_vecs)
        # print(A_vecs.size())
        A_vecs = torch.divide(A_vecs, torch.norm(A_vecs, p=2, dim=1, keepdim=True))
        B_vecs = torch.divide(B_vecs, torch.norm(B_vecs, p=2, dim=1, keepdim=True))
        matrix_logits = torch.matmul(A_vecs, torch.transpose(B_vecs, 0, 1)) * temperature # [batch, batch] each row represents one A item match all B
        tempa = matrix_logits.detach().cpu().numpy()
        # print("logits,", tempa.max(), tempa.min())
        matrix_softmax = torch.softmax(matrix_logits, dim=1) # softmax by dim=1
        tempb = matrix_softmax.detach().cpu().numpy()
        # print(np.diag(tempb))
        # print("softmax,", tempb.max(), tempb.min())
        matrix_log = -1 * torch.log(matrix_softmax)
        # here just use dig part
        loss_nce = torch.mean(torch.diag(matrix_log))
        return loss_nce
    
    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits, A_vecs, B_vecs = outputs
        # print(A_vecs.shape, B_vecs.shape)
        
        weight_nce = 1
        loss_nce_1 = self.infoNCE_diag(A_vecs, B_vecs) * weight_nce
        # loss_nce_2 = self.infoNCE(A_vecs, B_vecs, target) * weight_nce
        loss_nce = loss_nce_1

        loss_main = nn.CrossEntropyLoss(ignore_index=-1)(logits, target) * (1 - weight_nce)

        print('nce=%s, main=%s, loss=%s' % (loss_nce.detach().cpu().numpy(), loss_main.detach().cpu().numpy(), (loss_nce + loss_main).detach().cpu().numpy()))

        return loss_nce + loss_main   


def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "cross_trainer":
        return CrossTransformerTrainer(params)
    if trainer_type == "conv1d":
        return Conv1dTrainer(params)
    if trainer_type == "conv2d":
        return Conv2dTrainer(params)
    if trainer_type == "conv3d":
        return Conv3dTrainer(params)
    if trainer_type == "svm":
        return SVMTrainer(params) 
    if trainer_type == "random_forest":
        return RandomForestTrainer(params)
    if trainer_type == "knn":
        return KNNTrainer(params)
    if trainer_type == "contra_cross_transformer":
        return ContraCrossTransformerTrainer(params)
    if trainer_type == "contra_conv3d":
        return ContraConv3dTrainer(params)

    assert Exception("Trainer not implemented!")

