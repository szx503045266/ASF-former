# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Model implementation of ASF-former.
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .token_transformer import ASF_R_Encoder, Token_transformer
from .token_performer import Token_performer
from .transformer_block import Mlp, Attention, Block, ConvBranch_HMCB, Fusion_adaptive
from .utils import load_21k_pretrained

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ASF_former_p_S': _cfg(),
    'ASF_former_p_B': _cfg()
}

class ASF_p_C_Encoder(nn.Module):
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
        x_conv = x_emb[:, :, self.attn_dim:]
        B, HW, C = x_conv.shape
        x_conv = x_conv.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))

        x_attn = self.attn(self.norm1(x_attn))
        #x_attn = self.attn(self.norm1(x_attn.transpose(1,2)).transpose(1,2))

        x_conv = self.conv_branch(x_conv)
        B, C, H, W = x_conv.shape
        x_conv = x_conv.reshape(B, C, H*W).transpose(1,2)

        #x = x + self.drop_path(self.fc2(self.split_ratio * x_attn + (1 - self.split_ratio) * x_conv))
        x_fuse, w_attn, w_conv = self.fuse(x_attn, x_conv)
        x = x + self.drop_path(self.fc2(x_fuse))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, w_attn, w_conv

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, token_dim=64, downsample_ratio=4, depth=2, num_heads_R=1, num_heads_C=6, 
                dpr=0.1, mlp_ratio=3., ASF=True, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.ASF = ASF
        if downsample_ratio == 4:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_chans, token_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(token_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(token_dim // 2, token_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(token_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(token_dim, token_dim, kernel_size=3, stride=1, padding=1)
            )
        elif downsample_ratio == 2:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_chans, token_dim, kernel_size=3, stride=2, padding=1, bias=False)
            )
        if ASF == True:
            block = ASF_p_C_Encoder
        else:
            block = Block
        self.blocks = nn.ModuleList([
            block(
                dim=token_dim, num_heads=num_heads_C, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x):
        if len(x.shape) == 3:
            B, HW, C = x.shape
            x = x.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        x = self.down_sample(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).transpose(1, 2)
        for i, blk in enumerate(self.blocks):
            if self.ASF == True:
                x, _, _ = blk(x)
            else:
                x = blk(x)
        return x

class ASF_former_p(nn.Module):
    def __init__(self, img_size=224, in_chans=3, stages=4, num_classes=1000, token_dim=[64,128,256,512],
                 downsample_ratio=[4,2,2,2], depth=[2,2,8,2], num_heads_R=[1,1,1,1], num_heads_C=[6,6,6,6],
                 mlp_ratio=4., ASF=[True,True,True,False], qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.stages = stages
        self.num_classes = num_classes
        self.token_dim = token_dim
        self.downsample_ratio = downsample_ratio
        self.depth = depth
        self.ASF = ASF

        depth = np.sum(self.depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else np.sum(self.depth[:i])
            Layers.append(
                BasicLayer(img_size, in_chans, self.token_dim[i], self.downsample_ratio[i], self.depth[i], num_heads_R[i], num_heads_C[i],
                    dpr[startDpr:self.depth[i]+startDpr], mlp_ratio, self.ASF[i], qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer)
            )
            img_size = img_size // self.downsample_ratio[i]
            in_chans = self.token_dim[i]
        self.layers = nn.ModuleList(Layers)

        # Classifier head
        self.head = nn.Linear(self.token_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token','pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.token_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.mean(x, 1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def ASF_former_p_S(pretrained=False, **kwargs):
    model = ASF_former_p(token_dim=[64, 128, 256, 512], depth=[3,4,12,5], num_heads_R=[1, 1, 2, 4], num_heads_C=[1, 2, 4, 8],
                         mlp_ratio=3., ASF=[True,True,True,True], **kwargs)
    model.default_cfg = default_cfgs['ASF_former_p_S']
    return model

@register_model
def ASF_former_p_B(pretrained=False, **kwargs):
    model = ASF_former_p(token_dim=[96, 192, 384, 768], depth=[3,4,18,5], num_heads_R=[1, 1, 2, 4], num_heads_C=[1, 2, 4, 8],
                         mlp_ratio=3., ASF=[True,True,True,False], **kwargs)
    model.default_cfg = default_cfgs['ASF_former_p_B']
    return model