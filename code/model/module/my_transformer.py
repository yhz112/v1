# from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
from .patch import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches
from .patch import ExtractImagePatches
import pdb
import math

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSepConv, self).__init__()

        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # 输出通道数等于输入通道数
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # groups=in_channels 表示每个通道独立卷积
            bias=bias
        )

        # 逐点卷积 (Pointwise Convolution)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数由逐点卷积决定
            kernel_size=1,  # 1x1 卷积
            stride=1,
            padding=0,
            dilation=1,
            groups=1,  # 标准卷积
            bias=bias
        )

    def forward(self, x):
        # 先进行深度卷积，再进行逐点卷积
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class RepConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)


        conv3x3 = nn.Sequential(
            DepthwiseSepConv(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
            # nn.Hardswish(inplace= True)
        )

        self.body = nn.Sequential(conv1x1, conv3x3)

    def forward(self, x):
        return self.body(x)


class DepthwiseSepAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        
        self.q_conv = nn.Sequential(RepConvBlock(dim, dim, bias=False))
        self.k_conv = nn.Sequential(RepConvBlock(dim, dim, bias=False))
        self.v_conv = nn.Sequential(RepConvBlock(dim, dim, bias=False))
        
        
        self.proj_conv = nn.Sequential(
            RepConvBlock(dim, dim, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W


        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = (
            q.reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        
        k = (
            k.reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = (
            v.reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        
        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        
        x = q.mul(kv)
        
        x = x.transpose(2, 3).reshape(B, C, H, W).contiguous()        
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x

class SpatialConvMLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features
        # self.hard_swish_1 = nn.Hardswish(inplace= True)
        # self.hard_swish_2 = nn.Hardswish(inplace= True)

        self.relu6_1 = nn.ReLU6(inplace=True)
        # self.relu6_2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x = x.flatten(2)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.relu6_1(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()
        x = self.relu6_1(x)

        return x    

class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
    ):
        super().__init__()


        self.attn1 = DepthwiseSepAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )



        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpatialConvMLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn1(x)
        x = x + self.mlp(x)

        return x
