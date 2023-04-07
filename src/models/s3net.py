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
        super(SSN, self).__init__(*args)
        # 构造的时候需要告知W H C K四个超参
        self.W,self.H,self.C,self.K=args[0],args[1],args[2],args[3]
        # patch size=W*H PCA channel=C classNum=K
        # 输入数据是channels,patch_size,patch_size，会先压成一维，用1DCNN
        self.SpecMod=nn.Sequential(
            nn.Conv1d(in_channels=self.W*self.H,out_channels=self.W*self.H*32,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*32),
            
            nn.Conv1d(in_channels=self.W*self.H*32,out_channels=self.W*self.H*32,kernel_size=2,padding=False,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*32),

            nn.Conv1d(in_channels=self.W*self.H*32,out_channels=self.W*self.H*32,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*32),

            nn.Conv1d(in_channels=self.W*self.H*32,out_channels=self.W*self.H*64,kernel_size=3,padding=True,stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*64)
        )
        self.FT=nn.Sequential(
            nn.Conv1d(in_channels=self.W*self.H*64,out_channels=self.W*self.H*64,kernel_size=self.C/2,stride=self.C/2,padding=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.W*self.H*64)
        )
        # 之后数据reshape成二维，做2DCNN
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

    def forward(self,x:torch.Tensor):
        # 对输入进来的x进行reshape，使其符合1DCNN的输入条件(batch size, channel, series length)
        batch_size=x.size(0)
        x=torch.transpose(x,(0,2,3,1))
        x=x.reshape(batch_size,-1,self.C)
        x=self.SpecMod(x)
        x=self.FT(x)
        # 这里x变为 batch_size，W*H*64，1
        x=x.reshape(batch_size,self.W,self.H,-1)
        x=self.SpatMod(x)
        # 变成K层的WH，然后要对每层进行相加，现在是K*W*H。还需要中心的一长串
        y=x[:,self.W/2,self.H/2]
        y=y.reshape(-1)
        x=torch.sum(torch.sum(x,dim=2,keep_dim=False),dim=1,keep_dim=False)
        return x,y # 这是2个长度为k的Tensor,x是GAP，y是中心向量，后续loss要使用

class S3Net(nn.Module):
    def __init__(self, *args):
        super(S3Net, self).__init__(*args)
        self.W1,self.H1,self.C1=args[0],args[1],args[2]
        self.W2,self.H2,self.C2=args[3],args[4],args[5]
        self.K=args[6]
        self.upperStruct=SSN(self.W1,self.H1,self.C1,self.K)# big patch
        self.lowerStruct=SSN(self.W2,self.H2,self.C2,self.K)# small patch

    def forward(self,data):
        # data:两部分，前面是原始的，后面是增强的
        # data两部分各自的维度：batch,channel,patch_size,patch_size
        ori,aug=data
        batch_size=ori.size(0)
        # 要把中心点处的拿出来
        g1,l1=self.upperStruct(ori)
        g2,l2=self.lowerStruct(aug)
        # g1 g2的维度是 [batch,K]
        g3=0.5*g1+0.5*g2
        return g1,g2,g3,l1,l2





        
        


        



        