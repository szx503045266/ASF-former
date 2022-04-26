# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Implementation of encoders in computation stage.
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block_conv(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.conv = ConvBranch_HMCB(in_features = dim, hidden_features = dim, out_features=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).transpose(1,2)

        x = x + self.drop_path(x)
        x = self.norm2(x.transpose(1,2))
        x = x.transpose(1,2)
        x = x + self.drop_path(self.mlp(x))
        return x, torch.zeros(B), torch.zeros(B)

class ConvBranch_HMCB(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        return x

class ConvBranch_PCM(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 3, padding=1),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(hidden_features, out_features, 3, padding=1),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ConvBranch_ResNet(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None):
        super().__init__()
        hidden_features = hidden_features // 2 or in_features // 2
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False)
            #nn.BatchNorm2d(out_features)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.SiLU(inplace=True)
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
                #nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        x = self.relu(self.shortcut(x) + self.conv1(x))
        return x

class Fusion_adaptive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inter_dim = max(dim, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, inter_dim, 1),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(inter_dim, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        initial = x + y
        feats = initial.transpose(1,2)
        gap = self.pool(feats).unsqueeze(3)
        atten = self.fc2(self.fc1(gap))
        w_conv = self.sigmoid(atten).squeeze(-1)

        result = initial + (1 - w_conv) * x + w_conv * y
        
        w_conv = w_conv.squeeze(-1)
        return result, 1 - w_conv, w_conv

class Fusion_learnablew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor([[0.],[0.]]), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x, y):
        initial = x + y
        w = self.softmax(self.w)
        result = initial + w[0] * x + w[1] * y
        return result

class Fusion_no_shortcut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inter_dim = max(dim, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, inter_dim, 1),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(inter_dim, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        initial = x + y
        feats = initial.transpose(1,2)
        gap = self.pool(feats).unsqueeze(3)
        atten = self.fc2(self.fc1(gap))
        w_conv = self.sigmoid(atten).squeeze(-1)

        result = (1 - w_conv) * x + w_conv * y
        return result

class ASF_C_Encoder(nn.Module):
    def __init__(self, dim, num_heads, expand_disabled=True, expand_ratio = 1.5,
                 split_ratio=0.5, conv_ratio=1., mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand_disabled = expand_disabled
        self.split_ratio = split_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if not expand_disabled:
            embed_dim = int(dim * expand_ratio)
            self.fc1 = nn.Linear(dim, embed_dim, 1)
        else:
            embed_dim = dim
        self.attn_dim = int(embed_dim * split_ratio)
        self.conv_dim = embed_dim - self.attn_dim

        self.norm1 = norm_layer(self.attn_dim)
        #self.norm1 = nn.BatchNorm1d(self.attn_dim)

        self.attn = Attention(self.attn_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        conv_hidden_dim = int(self.conv_dim * conv_ratio)
        self.conv_branch = ConvBranch_HMCB(in_features = self.conv_dim, hidden_features = conv_hidden_dim, out_features=self.attn_dim)
        self.fuse = Fusion_adaptive(self.attn_dim)
        self.fc2 = nn.Linear(self.attn_dim, dim, 1)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if not self.expand_disabled:
            x_emb = self.fc1(x)
        else:
            x_emb = x
        x_attn = x_emb[:, :, 0:self.attn_dim]
        x_cls = x_emb[:, 0, self.attn_dim:].unsqueeze(1)
        x_conv = x_emb[:, 1:, self.attn_dim:]
        B, HW, C = x_conv.shape
        x_conv = x_conv.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))

        x_attn = self.attn(self.norm1(x_attn))
        #x_attn = self.attn(self.norm1(x_attn.transpose(1,2)).transpose(1,2))

        x_conv = self.conv_branch(x_conv)
        B, C, H, W = x_conv.shape
        x_conv = x_conv.reshape(B, C, H*W).transpose(1,2)
        x_conv = torch.cat((x_cls, x_conv), dim=1)

        #x = x + self.drop_path(self.fc2(self.split_ratio * x_attn + (1 - self.split_ratio) * x_conv))
        x_fuse, w_attn, w_conv = self.fuse(x_attn, x_conv)
        x = x + self.drop_path(self.fc2(x_fuse))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, w_attn, w_conv

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
