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
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer, Token_CNN, ASF_R_Encoder
from .token_performer import Token_performer
from .transformer_block import Block, Block_conv, ASF_C_Encoder, get_sinusoid_encoding
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
    'ASF_former_S': _cfg(),
    'ASF_former_B': _cfg(),
    'HMCB_former_S': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_24': _cfg()
}

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64, ASF=False, conv_only=False):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            if ASF == True:
                self.attention1 = ASF_R_Encoder(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
                self.attention2 = ASF_R_Encoder(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            elif conv_only == True:
                self.attention1 = Token_CNN(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
                self.attention2 = Token_CNN(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            else:
                self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
                self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        w_attn_depth = []
        w_conv_depth = []

        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x, w_attn, w_conv = self.attention1(x)
        w_attn_depth.append(w_attn.mean())
        w_conv_depth.append(w_conv.mean())
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x, w_attn, w_conv = self.attention2(x)
        w_attn_depth.append(w_attn.mean())
        w_conv_depth.append(w_conv.mean())
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x, w_attn_depth, w_conv_depth

class ASF_former(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, ASF=False, conv_init=False, conv_only=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.conv_init = conv_init
        self.conv_only = conv_only
        self.depth = depth

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim, ASF=ASF, conv_only=conv_only)
        num_patches = self.tokens_to_token.num_patches

        if conv_only:
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if ASF == True:
            block = ASF_C_Encoder
        elif conv_only == True:
            block = Block_conv
            norm_layer = nn.BatchNorm1d
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            block = Block
        self.blocks = nn.ModuleList([
            block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if not conv_only:
            trunc_normal_(self.cls_token, std=.02)
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
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        B = x.shape[0]
        x, w_attn_depth, w_conv_depth = self.tokens_to_token(x)

        if self.conv_only:
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                x, w_attn, w_conv = blk(x)
                w_attn_depth.append(w_attn.mean())
                w_conv_depth.append(w_conv.mean())
                if i == 0:
                    w_attn_class = w_attn
                    w_conv_class = w_conv
            
            x = self.norm(x.transpose(1,2))
            x = self.pool(x)
            return x.transpose(1,2).squeeze(1), torch.Tensor(w_attn_depth), torch.Tensor(w_conv_depth), w_attn_class, w_conv_class
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                x, w_attn, w_conv = blk(x)
                w_attn_depth.append(w_attn.mean())
                w_conv_depth.append(w_conv.mean())
                if i == 0:
                    w_attn_class = w_attn
                    w_conv_class = w_conv

            x = self.norm(x)
            return x[:, 0], torch.Tensor(w_attn_depth), torch.Tensor(w_conv_depth), w_attn_class, w_conv_class

    def forward(self, x):
        x, w_attn_depth, w_conv_depth, w_attn_class, w_conv_class = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def ASF_former_S(pretrained=False, **kwargs):
    model = ASF_former(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., ASF=True, conv_init=True, **kwargs)
    model.default_cfg = default_cfgs['ASF_former_S']
    return model

@register_model
def ASF_former_B(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = ASF_former(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., ASF=True, conv_init=True, **kwargs)
    model.default_cfg = default_cfgs['ASF_former_B']
    if pretrained:
        state_dict = load_21k_pretrained(
            pretrained, model, use_ema=True, num_classes=model.num_classes)
        model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def HMCB_former_S(pretrained=False, **kwargs):
    model = ASF_former(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., conv_only=True, conv_init=True, **kwargs)
    model.default_cfg = default_cfgs['HMCB_former_S']
    return model

@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = ASF_former(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_24(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = ASF_former(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = ASF_former(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = ASF_former(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model