# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Implementation of encoders in reduction stage.
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from .transformer_block import Mlp, ConvBranch_HMCB, Fusion_adaptive

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Token_CNN(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.conv = ConvBranch_HMCB(in_features = dim, hidden_features = dim, out_features=in_dim)    ## Here out_features != self.conv_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W)

        x = self.norm2(x)
        x = x.transpose(1,2)
        x = x + self.drop_path(self.mlp(x))
        return x, torch.zeros(B), torch.zeros(B)

class ASF_R_Encoder(nn.Module):
    def __init__(self, dim, in_dim, num_heads, expand_disabled=True, expand_ratio = 1.5,
                 split_ratio=0.5, conv_ratio=1., mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand_disabled = expand_disabled
        self.split_ratio = split_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if not expand_disabled:
            embed_dim = int(dim * expand_ratio)
            self.fc1 = nn.Linear(dim, embed_dim)
        else:
            embed_dim = dim
        self.attn_dim = int(embed_dim * split_ratio)
        self.conv_dim = embed_dim - self.attn_dim

        self.norm1 = norm_layer(self.attn_dim)
        #self.norm1 = nn.BatchNorm1d(self.attn_dim)

        self.attn = Attention(dim=self.attn_dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        conv_hidden_dim = int(self.conv_dim * conv_ratio)
        self.conv_branch = ConvBranch_HMCB(in_features = self.conv_dim, hidden_features = conv_hidden_dim, out_features=in_dim)    ## Here out_features != self.conv_dim
        self.fuse = Fusion_adaptive(in_dim)
        #self.fc2 = nn.Linear(in_dim, dim)

        self.norm2 = norm_layer(in_dim)
        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        if not self.expand_disabled:
            x_emb = self.fc1(x)
        else:
            x_emb = x
        x_attn = x_emb[:, :, 0:self.attn_dim]
        x_conv = x_emb[:, :, self.attn_dim:]
        B, HW, C = x_conv.shape
        x_conv = x_conv.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        
        x_attn = self.attn(self.norm1(x_attn))
        #x_attn = self.attn(self.norm1(x_attn.transpose(1,2)).transpose(1,2))

        x_conv = self.conv_branch(x_conv)
        B, C, H, W = x_conv.shape
        x_conv = x_conv.reshape(B, C, H*W).transpose(1,2)

        #x = self.split_ratio * x_attn + (1 - self.split_ratio) * x_conv
        x, w_attn, w_conv = self.fuse(x_attn, x_conv)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, w_attn, w_conv