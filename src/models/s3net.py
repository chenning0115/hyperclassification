import PIL
import time, json, collections, math
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat

class SSN(nn.Module):
    def __init__(self, *args):
        super(SSN, self).__init__(*args))
        self.W,self.H,self.C,self.K=args[0],args[1],args[2],args[3]
        # patch size=W*H PCA channel=C classNum=K
        # 目前的输入是已经做好patch和PCA的mini-batch，四维数据
        # 先是四个一维卷积核，针对channel维的.
        self.SpecMod=nn.Sequential(
            nn.Conv1d(in_channels=self.W*self.H,out_channels=self.W*self.H*32,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*32)
            
            nn.Conv1d(in_channels=self.W*self.H*32,out_channels=self.W*self.H*32,kernel_size=2,padding=False,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*32)

            nn.Conv1d(in_channels=self.W*self.H*32,out_channels=self.W*self.H*32,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*32)

            nn.Conv1d(in_channels=self.W*self.H*32,out_channels=self.W*self.H*64,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*64)
        )
        self.FT=nn.Sequential(
            nn.Conv1d(in_channels=self.W*self.H*64,out_channels=self.W*self.H*64,kernel_size=self.C/2,stride=self.C/2,padding=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*64)
        )
        self.SpatMod=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32,out_channels=self.K,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.K),
        )
        # GAP是将W*H*K，每一个大小为W*H的层的数字全部加起来，成为K维.这块不太会，只在forward中进行一个sum
        # self.GAP=torch.sum(torch.sum())

    def forward(self,x:Tensor):
        # 对输入进来的x进行reshape，使其符合1DCNN的输入条件(batch size, channel, series length)
        batch_size=s.size(0)
        x=x.reshape(batch_size,self.W*self.H,-1)
        x=self.SpecMod(x)
        x=self.FT(x)
        x=x.reshape(batch_size,self.W,self.H,-1)
        x=self.SpatMod(x)
        x=toch.sum(torch.sum(x,dim=1,keep_dim=False),dim=1,keep_dim=False)
        return x

class S3Net(nn.Module):
    def __init__(self, *args):
        super(S3Net, self).__init__(*args))
        self.W1,self.H1,self.C1=args[0],args[1],args[2]
        self.W2,self.H2,self.C2=args[3],args[4],args[5]
        self.upperStruct=SSN(self.W1,self.H1,self.C1)# big patch
        self.lowerStruct=SSN(self.W2,self.H2,self.C2)# small patch

    def forward(self,x:Tensor):
        batch_size=x.size(0)
        x=nn.LayerNorm(x.size()[1:])(x)
        H=x.size(1)
        W=x.size(2)
        C=x.size(3)
        x=x.reshape(batch_size,H*W,-1)
        # 做PCA并且划分两个patch
        pca=Tensor(batch_size,W,H,10)# 保留PCA前10维
        for i in range(batch_size):# x[i]为当前HSI图片像素flatten后的情况，dim0是像素，dim1是band
            cov=torch.cov(x[i])
            (evals,evecs)=torch.eig(cov,eigenvectors=True)
            order=evals.argsort()[::-1]
            evals=evals[order]
            evecs=evecs[:,order]
            pc=torch.matmul(x[i],evecs)[:,:,10]
            pc=pc.reshape(W,H,10)
            pca[i]=pc
        g1=self.upperStruct(pca)
        g2=self.lowerStruct(pca)
        g3=0.5*g1+0.5*g2
        return g1,g2,g3





        
        


        



        