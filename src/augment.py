import torch
from torchvision import transforms
'''
这个是对原patch进行缩小，参数size小于原patch的size
'''
class ShrinkAugment:
    def __init__(self,params) -> None:
        self.size=params.get("size",3)

    def do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        center=int(data.size(2)/2)
        margin=int((self.size-1)/2)
        newdata=torch.zeros(batch_size,channel_num,self.size,self.size)
        for i in range(batch_size):
            newdata[i]=data[i,:,center-margin:center+margin+1]
        return data,newdata

'''
使用高斯核对每个spectrum进行模糊，参数包括kernel_size和sigma_square
在json中：
"type":"Gauss"，
"kernel_size":5
"sigma_sq":2.25
'''
class GaussAugment:
    def __init__(self,params) -> None:
        self.kernel_size=params.get("kernel_size",3)
        self.sigma_sq=params.get("sigma_sq",2.25)

    def do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        newdata=torch.zeros(data.shape())
        t=transforms.GaussianBlur(self.kernel_size,self.sigma_sq)
        for i in range(batch_size):
            newdata[i]=t(data[i])
        return data,newdata

'''
使用在spectrum维的gaussblur
"type":"SpectralFilter"，
"kernel_size":5
"sigma_sq":2.25
'''
class SpecFilterAugment:
    def __init__(self,params) -> None:
        self.kernel_size=params.get("kernel_size",3)
        self.sigma_sq=params.get("sigma_sq",2.25)
        self.margin=self.kernel_size/2
        self.filter=torch.Tensor(self.kernel_size)
        for i in range(self.margin+1):
            self.filter[i]=self.filter[self.kernel_size-1-i]=-1*torch.exp((self.margin-i)*(self.margin-i)/2/self.sigma_sq)/torch.sqrt(2*torch.PI*self.sigma_sq)

    def do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        H=data.size(2)
        W=data.size(3)
        data=torch.transpose(data,(0,2,3,1))
        newdata=torch.zeros(data.shape())
        for i in range(batch_size):
            padding_data=torch.zeros(H,W,channel_num+2*self.margin)
            padding_data[:,:,self.margin:self.margin+channel_num+1]=data[i]
            for j in range(H):
                for k in range(W):
                    for l in range(channel_num):
                        newdata[i][j][k][l]=torch.dot(self.filter,padding_data[j][k][l:l+self.kernel_size])
        data=torch.transpose(data,(0,3,1,2))
        newdata=torch.transpose(newdata,(0,3,1,2))
        return data,newdata


def do_augment(params,data):# 增强也有一系列参数呢，比如multiscale的尺寸、mask的大小、Gaussian噪声的参数等
    if params['type']=='shrink':
        return ShrinkAugment(params).do(data)
    if params['type']=='Gauss':
        return GaussAugment(params).do(data)