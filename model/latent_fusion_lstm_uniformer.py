# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
import numpy as np

layer_scale = False
init_value = 1e-6


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


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_visualization = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if return_visualization:
            attn_copy = attn.clone().detach().cpu().numpy()
            attn_copy = np.sum(attn_copy,axis=2) # N,head_num,Z*H*W


        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_visualization:
            return x, attn_copy
        return x, None


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, bn = 0):
        super().__init__()
        self.bn = bn
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x, return_visualization = False, fusion = False, D_ = None, H_ = None, W_ = None, handcraft_branch=False):
        if fusion:
            assert D_ is not None and H_ is not None and W_ is not None
            B,N,C = x.shape 
            if handcraft_branch:
                bottleneck = x[:,-(self.bn+1):,:]
                x = x[:,:-(self.bn+1),:]
            else:
                bottleneck = x[:,-self.bn:,:]
                x = x[:, :-self.bn, :] if self.bn != 0 else x
            x = x.permute(0,2,1).contiguous().reshape(B,C,D_,H_,W_)
            x = x + self.pos_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([x,bottleneck],dim=1) if self.bn != 0 else x
            if self.ls:
                x_attn, visualization_heads = self.attn(self.norm1(x),return_visualization)
                x_attn = self.drop_path(self.gamma_1 * x_attn)
                x = x_attn + x
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x_attn, visualization_heads = self.attn(self.norm1(x),return_visualization)
                x_attn = self.drop_path(x_attn)
                x = x_attn + x
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.pos_embed(x)
            B, C, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            if self.ls:
                x_attn, visualization_heads = self.attn(self.norm1(x),return_visualization)
                x_attn = self.drop_path(self.gamma_1 * x_attn)
                x = x_attn + x
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x_attn, visualization_heads = self.attn(self.norm1(x),return_visualization)
                x_attn = self.drop_path(x_attn)
                x = x_attn + x
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x, visualization_heads
   

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

class ResBiLstm(nn.Module):
    def __init__(self,hidden_size,drop=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.Bi_lstm_layer1 = nn.LSTM(hidden_size,hidden_size//2,bidirectional=True,batch_first=True)
        self.Bi_lstm_layer2 = nn.LSTM(hidden_size,hidden_size//2,bidirectional=True,batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop)
    
    def forward(self,x):
        x = self.layer_norm(self.Bi_lstm_layer1(x)[0])+x
        x = self.dropout(x)
        x = self.layer_norm(self.Bi_lstm_layer2(x)[0])+x
        
        return x

class Latent_fusion_Lstm_UniFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[3, 4, 8, 3], img_size=224, split_phase_num = 4, split_in_chans = [2,2,4,1], num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, conv_stem=False, pretrained_cfg_overlay=None, return_visualization = False,
                 return_hidden = False, handcraft_branch=None, bottleneck_n = None):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        self.split_phase_num = split_phase_num
        self.split_in_chans = split_in_chans
        self.depth = depth
        self.handcraft_branch = handcraft_branch

        self.patch_embed1 = []
        self.patch_embed2 = []
        self.patch_embed3 = []
        self.patch_embed4 = []

        if conv_stem:
            for i in range(split_phase_num):
                self.patch_embed1.append(head_embedding(in_channels=split_in_chans[i], out_channels=embed_dim[0]))
                self.patch_embed2.append(middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1]))
                self.patch_embed3.append(middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2], stride=(1, 2, 2)))
                self.patch_embed4.append(middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3], stride=(1, 2, 2)))

        else:
            for i in range(split_phase_num):
                self.patch_embed1.append(PatchEmbed(
                    img_size=img_size, patch_size=(1, 2, 2), in_chans=split_in_chans[i], embed_dim=embed_dim[0]))
                self.patch_embed2.append(PatchEmbed(
                    img_size=img_size // 4, patch_size=(1, 2, 2), in_chans=embed_dim[0], embed_dim=embed_dim[1]))
                self.patch_embed3.append(PatchEmbed(
                    img_size=img_size // 8, patch_size=(1, 2, 2), in_chans=embed_dim[1], embed_dim=embed_dim[2], stride=(1, 2, 2)))
                self.patch_embed4.append(PatchEmbed(
                    img_size=img_size // 16, patch_size=(1, 2, 2), in_chans=embed_dim[2], embed_dim=embed_dim[3], stride=(1, 2, 2)))
        
        self.patch_embed1 = nn.ModuleList(self.patch_embed1)
        self.patch_embed2 = nn.ModuleList(self.patch_embed2)
        self.patch_embed3 = nn.ModuleList(self.patch_embed3)
        self.patch_embed4 = nn.ModuleList(self.patch_embed4)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        self.blocks4 = []

        # init blocks1
        for l in range(depth[0]):
            self.blocks1.append([])
            for i in range(split_phase_num):
                self.blocks1[l].append(CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))
            self.blocks1[l] = nn.ModuleList(self.blocks1[l])
        self.blocks1 = nn.ModuleList(self.blocks1)

        for l in range(depth[1]):
            self.blocks2.append([])
            for i in range(split_phase_num):
                self.blocks2[l].append(CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer))
            self.blocks2[l] = nn.ModuleList(self.blocks2[l])
        self.blocks2 = nn.ModuleList(self.blocks2)

        for l in range(depth[2]):
            self.blocks3.append([])
            for i in range(split_phase_num):
                self.blocks3[l].append(SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer))
            self.blocks3[l] = nn.ModuleList(self.blocks3[l])
        self.blocks3 = nn.ModuleList(self.blocks3)

        for l in range(depth[3]):
            self.blocks4.append([])
            for i in range(split_phase_num-1):
                self.blocks4[l].append(SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer,bn=bottleneck_n))
            self.blocks4[l].append(SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer))
            self.blocks4[l] = nn.ModuleList(self.blocks4[l])
        self.blocks4 = nn.ModuleList(self.blocks4)
        
        self.norm = []
        for i in range(split_phase_num):
            self.norm.append(nn.BatchNorm3d(embed_dim[-1]))
        self.norm = nn.ModuleList(self.norm)
        
        self.reslstm_layer = ResBiLstm(embed_dim[3],drop=drop_rate)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.return_visualization = return_visualization
        self.return_hidden = return_hidden
        
        # init cross phase tokens
        if bottleneck_n is not None:
            # remember tile (bs,1,1) when forward
            self.bottleneck = nn.Parameter(torch.Tensor(1, bottleneck_n, embed_dim[-1]))
            nn.init.normal_(self.bottleneck, std=.02)
        else:
            self.bottleneck = None
        
        if self.handcraft_branch:
            self.handcraft_extractor = []
            for i in range(split_phase_num):
                self.handcraft_extractor.append(nn.Sequential(nn.Linear(93,embed_dim[-1]),
                                                    nn.ReLU(),
                                                    nn.Linear(embed_dim[-1],embed_dim[-1])
                                                    )) 
            self.handcraft_extractor = nn.ModuleList(self.handcraft_extractor)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # if isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, handcraft_input = None):
        N,T,Z,H,W = x.shape
        x_split = list(torch.split(x,self.split_in_chans[:-1],dim=1))
        x_split_time_branch = x_split[-1].reshape(-1,1,Z,H,W).contiguous()
        x_split.append(x_split_time_branch)

        if handcraft_input is not None:
            handcraft_input_split = list(torch.split(handcraft_input,self.split_in_chans[:-1],dim=1))
            handcraft_input_split_time_branch = handcraft_input_split[-1].reshape(-1,1,handcraft_input_split[-1].shape[-1]).contiguous()
            handcraft_input_split.append(handcraft_input_split_time_branch)
            for i in range(self.split_phase_num):
                handcraft_input_split[i] = self.handcraft_extractor[i](handcraft_input_split[i])
                handcraft_input_split[i] = torch.mean(handcraft_input_split[i],dim=1)

        # patch embed 1 & blocks1 
        for i in range(self.split_phase_num):
            x_split[i] = self.patch_embed1[i](x_split[i])
            x_split[i] = self.pos_drop(x_split[i])

        for l in range(self.depth[0]):
            for i in range(self.split_phase_num):
                x_split[i] = self.blocks1[l][i](x_split[i])

        # patch embed 2 & blocks2
        for i in range(self.split_phase_num):
            x_split[i] = self.patch_embed2[i](x_split[i])

        for l in range(self.depth[1]):
            for i in range(self.split_phase_num):
                x_split[i] = self.blocks2[l][i](x_split[i])

        # patch embed 3 & blocks3
        for i in range(self.split_phase_num):
            x_split[i] = self.patch_embed3[i](x_split[i])

        for l in range(self.depth[2]):
            for i in range(self.split_phase_num):
                x_split[i],_ = self.blocks3[l][i](x_split[i])

        # patch embed 4 & blocks4
        for i in range(self.split_phase_num):
            x_split[i] = self.patch_embed4[i](x_split[i])

        # cross phase tokens
        if self.bottleneck is not None:
            batch_bottleneck = self.bottleneck.expand(N, -1, -1)
        else:
            batch_bottleneck = None
        
        visualization_heads_list = []
        for l in range(self.depth[3]):
            bottle = []
            for i in range(self.split_phase_num-1):
                N_,C_,Z_,H_,W_ = x_split[i].shape
                x_split[i] = x_split[i].flatten(2).permute(0,2,1).contiguous() # N,C,T -> N,T,C
                if self.handcraft_branch:
                    x_split[i] = torch.cat([x_split[i],handcraft_input_split[i].unsqueeze(1)],dim=1)
                t_split = x_split[i].shape[1]
                in_mod = torch.cat([x_split[i],batch_bottleneck],dim=1)
                out_mod, visualization_heads = self.blocks4[l][i](in_mod,return_visualization = self.return_visualization,fusion=True, D_ = Z_, H_ = H_, W_ = W_, handcraft_branch=self.handcraft_branch)
                if self.return_visualization and l == self.depth[3]-1:
                    visualization_heads = visualization_heads[:, :, :t_split]
                    if self.handcraft_branch:
                        visualization_heads = visualization_heads[:, :, :-1]
                    visualization_heads_list.append(visualization_heads)
                x_split[i] = out_mod[:, :t_split, ...]
                if self.handcraft_branch:
                    x_split[i] = x_split[i][:,:-1,:]
                x_split[i] = x_split[i].permute(0,2,1).contiguous().reshape(N_,C_,Z_,H_,W_)
                bottle.append(out_mod[:, t_split:, ...])
            batch_bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
        
        for l in range(self.depth[3]):
            x_split[-1],visualization_heads = self.blocks4[l][-1](x_split[-1],return_visualization = self.return_visualization)
            if self.return_visualization and l == self.depth[3]-1:
                visualization_heads_list.append(visualization_heads)

        if self.return_visualization:
            depth, height, width = Z_,H_,W_
            for idx,item in enumerate(visualization_heads_list):
                visualization_heads_list[idx] = item.reshape((item.shape[0],item.shape[1],depth, height, width))

        batch_bottleneck = batch_bottleneck.permute(0,2,1).contiguous()

        for i in range(self.split_phase_num):
            x_split[i] = self.norm[i](x_split[i])
            x_split[i] = x_split[i].flatten(2).mean(-1)
        
        if self.handcraft_branch:
            x_split[-1] = x_split[-1]+handcraft_input_split[-1]
        x_split[-1] = x_split[-1].reshape(N,-1,self.embed_dim[-1])
        x_split[-1] = x_split[-1] + self.reslstm_layer(x_split[-1])
        x_split[-1] = torch.mean(x_split[-1],dim=1)
        
        for i in range(self.split_phase_num):
            x_split[i] = self.pre_logits(x_split[i])

        return x_split,visualization_heads_list

    def forward(self, x, handcraft_input=None):

        x, visualization_heads = self.forward_features(x, handcraft_input)
        
        x_out = []
        for i in range(self.split_phase_num):
            x_out.append(x[i])
        
        x_out = torch.stack(x_out,dim=1) # N,phase_num,C
        x_out = torch.mean(x_out,dim=1)

        if self.return_hidden:
            return x_out, visualization_heads

        x_out = self.head(x_out)
        return x_out, visualization_heads

@register_model
def latent_fusion_lstm_uniformer_base(num_classes=2, 
                       split_phase_num = 4, 
                       split_in_chans = [2,2,4,1],
                       img_size = 224,
                       patch_size = (2,2,2),
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = Latent_fusion_Lstm_UniFormer(
        split_phase_num=split_phase_num,
        split_in_chans=split_in_chans,
        num_classes=num_classes,
        img_size = img_size,
        bottleneck_n=4, # modify
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwards)
    for i in range(split_phase_num):
        model.patch_embed1[i] = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=split_in_chans[i], embed_dim=model.embed_dim[0])
    
    model.default_cfg = _cfg()
    if pretrained:
        default_pretrained_path = os.path.join('./pretrained_model/uniformer_base_k600_16x8.pth')
        model_para = torch.load(default_pretrained_path)

        for para_name in list(model_para.keys()):
            if 'patch_embed1' in para_name:
                model_para.pop(para_name)
            if 'head' in para_name:
                model_para.pop(para_name)

        keep_state = {}
        for k,v in model_para.items():
            if "patch_embed" in k:
                for i in range(split_phase_num):
                    q_k = k.split('.')
                    q_k.insert(1, '%s'%i)
                    q_k = ".".join(q_k)
                    if q_k in model.state_dict().keys():
                        keep_state[q_k] = v
            elif "blocks" in k:
                for i in range(split_phase_num):
                    q_k = k.split('.')
                    q_k.insert(2, '%s'%i)
                    q_k = ".".join(q_k)
                    if q_k in model.state_dict().keys():
                        keep_state[q_k] = v
            elif "norm" in k:
                for i in range(split_phase_num):
                    q_k = k.split('.')
                    q_k.insert(1, '%s'%i)
                    q_k = ".".join(q_k)
                    if q_k in model.state_dict().keys():
                        keep_state[q_k] = v

        missing_state = set(list(model.state_dict().keys()))-set(list(keep_state.keys()))

        cur_model_state = model.state_dict()
        cur_model_state.update(keep_state)

        model.load_state_dict(cur_model_state)

        # model.load_state_dict(model_para,strict=False)

    return model
