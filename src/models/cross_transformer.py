import PIL
import time, json
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, dim, heads=8, droupout=0.1) -> None:
        '''
            给定q和kv, 利用attention的方式返回对q的新空间编码new_q
            其中q的输入维度为(batch, seq, q_dim), 最终输出维度为(batch, seq, dim)
        '''
        super().__init__()
        self.heads = heads
        self.scale = kv_dim ** -0.5 #1/sqrt(dim)

        self.to_q = nn.Linear(q_dim, dim, bias=True) # dim = heads * per_dim
        self.to_k = nn.Linear(kv_dim, dim, bias=True)
        self.to_v = nn.Linear(kv_dim, dim, bias=True)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(droupout)



    def forward(self, x, y, mask=None):
        # x shape is (batch, seq1, q_dim)
        # y shape is (batch, seq2, kv_dim)
        b, n, _, h = *x.shape, self.heads
        by, ny, _, hy= *x.shape, self.heads
        assert b == by

        # q,k,v获取
        qheads, kheads, vheads = self.to_q(x), self.to_k(y), self.to_v(y) # qheads,kvheads shape all is (batch, seq, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (qheads, kheads, vheads))  # split into multi head attentions
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out) # (batch, seq1, dim)
        return out

class Attention(nn.Module):

    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, drouput) -> None:
        '''
            输入x和y, 将x在y空间中进行transformer的encoder生成x的新表示 new_x
            输入x和y, 将y在x空间中进行transformer的encoder生成y的新表示 new_y
            输入x和y维度应该相同 否则无法做residule 输入dim,输出dim
            x shape(batch, seq1, dim)
            y shape(batch, seq2, dim)
        '''
        super().__init__()
        # CrossTransformer 目前只支持one layer, 即depth=1
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_x2 = nn.LayerNorm(dim)
        self.norm_y2 = nn.LayerNorm(dim)

        self.cross_attention_x = CrossAttention(dim, dim, dim, heads=heads, droupout=drouput)
        self.cross_attention_y = CrossAttention(dim, dim, dim, heads=heads, droupout=drouput)

        self.mlp_x = MLP_Block(dim, mlp_dim, dropout=drouput)
        self.mlp_y = MLP_Block(dim, mlp_dim, dropout=drouput)

    def forward(self, x, y, mask=None):
        assert mask==None
        # x和y会分别作为q以及对应的kv进行cross-transformer
        #1. 保留shortcut
        shortcut_x = x
        shortcut_y = y

        #2. 做prenorm
        x = self.norm_x(x)
        y = self.norm_y(y)

        #3. 分别做cross-attention
        x = self.cross_attention_x(x, y, mask=mask)
        y = self.cross_attention_y(y, x, mask=mask)

        #4. short cut收
        x = shortcut_x + x
        y = shortcut_y + y

        #5. 做mlp 和 residual
        x = x + self.mlp_x(self.norm_x2(x))
        y = y + self.mlp_y(self.norm_y2(y))

        return x, y



class HSINet(nn.Module):
    def __init__(self, params):
        super(HSINet, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        num_classes = data_params.get("num_classes", 16)
        patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)

        dim = net_params.get("dim", 64)
        depth = net_params.get("depth", 5)
        heads = net_params.get("heads", 4)
        dim_heads = net_params.get("dim_heads", 16)
        mlp_dim = net_params.get("mlp_dim", 8)
        dropout = net_params.get("dropout", 0.)
        
        image_size = patch_size * patch_size

        # 依据光谱连续性降低spectral维度 同时引入空间信息 200-3+2*2
        conv3d_kernal_size = net_params.get("conv3d_kernal_size", [3,5,5])
        conv3d_stride = net_params.get("conv3d_stride", [3,1,1])
        conv3d_padding = net_params.get("conv3d_padding", [2,2,2])
        self.conv3d_for_spectral_trans = nn.Sequential(
            nn.Conv3d(1, out_channels=1, kernel_size=conv3d_kernal_size, stride=conv3d_stride, padding=conv3d_padding),
            nn.ReLU(),
        )

        self.new_spectral_size = int((self.spectral_size - conv3d_kernal_size[0] + 2 * conv3d_padding[0]) / conv3d_stride[0]) + 1
        self.new_image_size = image_size
        print("new_spectral_size", self.new_spectral_size)
        print("new_image_size", self.new_image_size)
        self.spectral_patch_embedding = nn.Linear(self.new_image_size, dim)
        self.pixel_patch_embedding = nn.Linear(self.spectral_size, dim)

        self.local_trans_spectral = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.local_trans_pixel = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout)

        self.cross_trans = CrossTransformer(dim=dim, heads=heads, mlp_dim=mlp_dim, drouput=dropout)

        self.spectral_pos_embedding = nn.Parameter(torch.randn(1, self.new_spectral_size+1, dim))
        self.pixel_pos_embedding = nn.Parameter(torch.randn(1, self.new_image_size+1, dim))

        mlp_head_dim = params['net'].get('mlp_head_dim', 8)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_head_dim),
            nn.Linear(mlp_head_dim, num_classes)
        )

        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_spectral = nn.Identity()
        self.to_latent_pixel = nn.Identity()

    def forward(self, x):
        '''
        x: (batch, p, w, h), s=spectral, w=weigth, h=height

        '''
        x_spectral = x_pixel = x 

        b, s, w, h = x_spectral.shape
        img = w * h
        #0. Conv
        # x_pixel = self.conv2d_for_pixel_trans(x) #(batch, p, w, h)
        x_spectral = torch.unsqueeze(x_spectral, 1) #(batch, c, p, w, h)
        x_spectral = self.conv3d_for_spectral_trans(x_spectral)
        x_spectral = torch.squeeze(x_spectral, 1) #(batch, p, w, h)

        #1. reshape
        x_spectral = rearrange(x_spectral, 'b s w h-> b s (w h)') # (batch, s, w*h)
        x_pixel = rearrange(x_pixel, 'b s w h-> b (w h) s') # (batch, s, w*h)

        #2. patch_embedding
        x_spectral = self.spectral_patch_embedding(x_spectral) #(batch, s`, dim)
        x_pixel = self.pixel_patch_embedding(x_pixel) #(batch, image_size, dim)

        #3. local transformer
        cls_tokens_spectral = repeat(self.cls_token_spectral, '() n d -> b n d', b = b) #[b,1,dim]
        x_spectral = torch.cat((cls_tokens_spectral, x_spectral), dim = 1) #[b,s`+1,dim]
        x_spectral = x_spectral + self.spectral_pos_embedding[:,:]

        cls_tokens_pixel = repeat(self.cls_token_pixel, '() n d -> b n d', b = b) #[b,1,dim]
        x_pixel = torch.cat((cls_tokens_pixel, x_pixel), dim = 1) #[b,image+1,dim]
        x_pixel = x_pixel + self.pixel_pos_embedding[:,:]

        x_spectral = self.local_trans_spectral(x_spectral) #(batch, s`+1, dim)
        x_pixel = self.local_trans_pixel(x_pixel) #(batch, image_size+1, dim)

        #4. cross transformer
        x_spectral_cross, x_pixel_cross = self.cross_trans(x_spectral, x_pixel) #(batch, s, dim), (batch, image_size, dim)

        #5. get spectral_pixel_feature
        # avgpooling for eatch trans_result to get the real feature map for mlp
        mean_spectral, mean_pixel = map(lambda temp : torch.mean(temp, dim=1)  , [x_spectral, x_pixel])
        x_spectral_cross, x_pixel_cross = map(lambda temp : torch.mean(temp, dim=1)  , [x_spectral_cross, x_pixel_cross])
        # x_feature = torch.concat([x_spectral, x_pixel, x_spectral_cross, x_pixel_cross], axis=1)

        # logits_x = self.mlp_head(x_feature)
        # logits_x = self.mlp_head(x_spectral)
        logit_spectral = self.to_latent_spectral(x_spectral[:,0])
        logit_pixel = self.to_latent_pixel(x_pixel[:,0])
        # logit_x = torch.concat([logit_spectral, logit_pixel, x_spectral_cross, x_pixel_cross], dim=-1)

        # --- TODO: just for experiment ---
        if self.params['net']['net_type'] == 'just_spectral':
            logit_x = torch.concat([logit_spectral], dim=-1)
        elif self.params['net']['net_type'] == 'just_pixel':
            logit_x = torch.concat([logit_pixel], dim=-1)
        elif self.params['net']['net_type'] == 'spectral_pixel':
            logit_x = torch.concat([logit_spectral, logit_pixel], dim=-1)
        elif self.params['net']['net_type'] == 'cross':
            logit_x = torch.concat([x_spectral_cross, x_pixel_cross], dim=-1)
        elif self.params['net']['net_type'] == 'spectral_pixel_cross':
            logit_x = torch.concat([logit_spectral, logit_pixel, x_spectral_cross, x_pixel_cross], dim=-1)
        elif self.params['net']['net_type'] == 'spectral_cross':
            logit_x = torch.concat([logit_spectral, x_spectral_cross, x_pixel_cross], dim=-1)
        else:
            raise Exception('cross wrong net type.')
        # --- TODO: just for experiment ---

        return  self.mlp_head(logit_x), logit_spectral, logit_pixel 
        # return  self.mlp_head(logit_x), mean_spectral, mean_pixel 

        
if __name__ == '__main__':
    path_param = './params/cross_param.json'
    with open(path_param, 'r') as fin:
        param = json.loads(fin.read())
    model = HSINet(param)
    model.eval()
    print(model)
    input = torch.randn(3, 200, 9, 9)
    y = model(input)
    print(y.shape)
