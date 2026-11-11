import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.transformer import TransBlock
from model.module.patch import reverse_patches, ReversePatches
from model.idwt import DWT, IWT

import numpy as np
from functools import partial
from .fan import FANLayer
from .module.my_transformer import MS_Block
from .u_net import UNet
from .module.mabma_s4 import LightMambaS4


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class BNHardSwish(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        # self.acti = nn.ReLU6(inplace = True)
        self.acti = nn.Hardswish(inplace= True)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class GatedFusion1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.midchannels = in_channels //2

        self.conv1x1_1 = nn.Sequential(
            ConvBn(in_channels, self.midchannels, kSize=1, stride=1, padding=0)
        )
        self.conv1x1_2 = nn.Sequential(
            ConvBn(in_channels, self.midchannels, kSize=1, stride=1, padding=0)
        )

        self.con3x3 = nn.Sequential(
            ConvBn(in_channels, in_channels, 
                   kSize=3, stride=1, padding=1)
        )
        self.bn_relu6 = BNPReLU(in_channels)

        
    def forward(self, high_feat, low_feat):
        high = self.conv1x1_1(high_feat)
        low = self.conv1x1_2(low_feat)
        fused_features = torch.cat([high, low], dim=1)  # [B, 3*C, H, W]
        x = self.con3x3(fused_features)

        return self.bn_relu6(x + fused_features)

class GatedFusion111(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.midchannels = max(in_channels*2 // 8, 8)


        self.conv1x1_2 = nn.Sequential(
            ConvBn(self.midchannels, in_channels, kSize=1, stride=1, padding=0)
        )

        self.con3x3 = nn.Sequential(
            ConvBn(in_channels * 2, self.midchannels, 
                   kSize=3, stride=1, padding=1)
        )
        self.conv1x1_3 = nn.Sequential(
            ConvNoBn(in_channels*2, in_channels, kSize=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        self.bn_relu6 = BNPReLU(in_channels)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, self.midchannels // 2, 1),  # 输入通道为拼接后的通道数
            nn.ReLU6(),
            nn.Conv2d(self.midchannels // 2, self.midchannels, 1),  # 输出通道调整为 mid_channels
            nn.Sigmoid()
        )

        
    def forward(self, high_feat, low_feat):
        fused_features = torch.cat([high_feat, low_feat], dim=1)  # [B, 3*C, H, W]
        res = self.conv1x1_3(fused_features)
        x = self.con3x3(fused_features)

        gate = self.attn(fused_features)
        x = self.conv1x1_2(x*gate)

        return self.bn_relu6(x + res)
        # return x

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.conv1x1_1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels = in_channels * 2,
        #         out_channels= in_channels * 2,  
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),
        #     BNPReLU(in_channels * 2)
        #     # ConvBn(in_channels, in_channels, kSize=3, stride=1, padding=1)
        # )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels * 2,
                out_channels= in_channels,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.BatchNorm2d(in_channels),
            # nn.Sigmoid()
            BNPReLU(in_channels)

            # ConvBn(in_channels, in_channels, kSize=3, stride=1, padding=1)
        )
        # self.conv3x3 = nn.Sequential(
        #     ConvBn(in_channels * 2, in_channels * 2, kSize=3, stride=1, padding=1)
        # )

        self.pa = PA(in_channels * 2)
        
    def forward(self, high_feat, low_feat):
        fused_features = torch.cat([high_feat, low_feat], dim=1) 
        # fused_features = self.conv3x3(fused_features)
        # fused_features = self.conv1x1_1(fused_features)
        # pa = self.pa(fused_features)
        fused_features = self.pa(fused_features)
        return self.conv1x1_2(fused_features)
        # return self.pa(high_feat + low_feat)


        # return x


class GatedFusion111(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv3x3 = ConvBn(in_channels*2, in_channels, kSize=3, stride=1, padding= 1)
        self.ec1 = eca_layer(in_channels)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BNPReLU(in_channels)
        )
        
    # def forward(self, input):
    def forward(self, high_feat, low_feat):

        output = self.conv3x3(torch.cat([high_feat , low_feat], dim=1))
        output = self.ec1(output)
        output = self.conv1x1(output)

        return output


class DABFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),  # 输出通道调整为 mid_channels
            BNPReLU(in_channels)
        )

        # self.attn = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels * 2, in_channels, 1),  # 输入通道为拼接后的通道数
        #     nn.ReLU6(),
        #     nn.Conv2d(in_channels, in_channels*2, 1),  # 输出通道调整为 mid_channels
        #     nn.Hardsigmoid()
        #     # nn.Sigmoid()
        # )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels * 2, 1),  # 输出通道调整为 mid_channels
            nn.Hardsigmoid(inplace= True)
            # nn.Sigmoid()
            # nn.Sigmoid()
        )
        # self.bnrelu6 = BNPReLU(in_channels)
        
    def forward(self, high_feat, low_feat):
        fused_features = torch.cat([high_feat, low_feat], dim=1)  # [B, 3*C, H, W]
        avg_out = self.avg_pool(fused_features)
        max_out = self.max_pool(fused_features)
        # combined = torch.cat([avg_out, max_out], dim=1)
        # combined = avg_out + max_out
        gate = self.attn(avg_out + max_out)

        # x = self.conv1x1_2(fused_features * gate) 
        return self.conv1x1_2(fused_features * gate)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 共享的全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        # 相加并通过 sigmoid
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3)  # 调整形状为 [B, C, 1, 1]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        # 拼接并卷积
        out = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        out = self.conv(out)  # [B, 1, H, W]
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)  # [B, C, H, W]
        # 空间注意力
        x = x * self.spatial_attention(x)  # [B, C, H, W]
        return x

class HybridAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = eca_layer(channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        return self.channel_att(x) * self.spatial_att(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,  # 输出通道数等于输入通道数
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,  # groups=in_channels 表示每个通道独立卷积
                bias=bias
            )
        )

        # 逐点卷积 (Pointwise Convolution)
        self.pointwise =  nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,  # 输出通道数由逐点卷积决定
                kernel_size=1,  # 1x1 卷积
                stride=1,
                padding=0,
                dilation=1,
                groups=1,  # 标准卷积
                bias=bias
            )
        )



    def forward(self, x):
        # 先进行深度卷积，再进行逐点卷积
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class ConvNoBn(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), bias=False):
        super().__init__()

        self.conv = DepthwiseSeparableConv(nIn, nOut, kernel_size=kSize,
                                           stride=stride, padding=padding, 
                                           dilation=dilation, bias=bias)



        # self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
        #                       stride=stride, padding=padding,
        #                       dilation=dilation, groups=1, bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvBn(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), bias=False):
        super().__init__()

        # self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
        #                       stride=stride, padding=padding,
        #                       dilation=dilation, groups=1, bias=bias)
                              
        self.conv = DepthwiseSeparableConv(nIn, nOut, kernel_size=kSize,
                                           stride=stride, padding=padding, 
                                           dilation=dilation,  bias=bias)

        self.bn_prelu = BNPReLU(nOut)
        # self.bn_hard_swish = BNHardSwish(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn_prelu(output)
        # output = self.bn_hard_swish(output)

        return output


class ChannelAttention(nn.Module):
    """通道注意力 (轻量化版)"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU6(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LightCoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向池化 (N,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向池化 (N,C,1,W)
        
        # 通道压缩（至少保留8通道）
        mip = max(8, inp // reduction)
        
        # 共享参数的压缩层
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace= True)
        
        # 独立分支的注意力生成
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x  # 残差连接保留原始特征
        
        # 步骤1：分解式空间编码
        x_h = self.pool_h(x)  # (N,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N,C,W,1)
        
        # 步骤2：通道压缩与融合
        y = torch.cat([x_h, x_w], dim=2)  # (N,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))
        
        # 步骤3：方向分离与注意力生成
        x_h, x_w = torch.split(y, [x_h.size(2), x_w.size(2)], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复维度 (N,C,1,W)
        
        # 步骤4：生成注意力权重
        a_h = self.sigmoid(self.conv_h(x_h))  # 水平注意力图 [0,1]
        a_w = self.sigmoid(self.conv_w(x_w))  # 垂直注意力图 [0,1]
        
        # 步骤5：残差增强
        return identity * a_w * a_h  # 位置敏感特征增强

class ATModule(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.mid_channels = nIn  # 保持通道一致



        # 动态多分支卷积
        self.conv_branch = nn.Sequential(
            nn.Conv2d(nIn, nIn*2, 3, padding=1, groups=nIn),  # 深度可分离卷积
            nn.Conv2d(nIn*2, nIn, 1),  # Pointwise卷积
            LightCoordAtt(nIn, nIn)
        )

        # 残差连接 + 动态激活
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(nIn, nOut, 1),
            nn.BatchNorm2d(nOut), 
            nn.LeakyReLU(0.1)
            # nn.Hardsigmoid() if nIn == nOut else nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = x
        x = self.conv_branch(x)
        # return self.conv1x1_out(x) + identity  # 残差融合
        return self.conv1x1_out(x + identity)  # 残差融合

class FTModule(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()

        self.conv_3x3 = nn.Sequential(
            ConvNoBn(nIn = nOut, nOut = nOut, kSize = 3, stride = 1, padding = 1),
            # nn.BatchNorm2d(nOut)
        )

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        # self.pa = PA(nOut)
        self.attn = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(nOut, nOut//2, 1),  # 输出通道调整为 mid_channels
            # nn.ReLU6(inplace= True),
            nn.Conv2d(nOut, nOut, 1),  # 输出通道调整为 mid_channels
            nn.Hardsigmoid(inplace= True)
        )

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(
                in_channels=nIn,
                out_channels=nOut,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BNPReLU(nOut)

            # ConvBn(nIn = nOut, nOut = nOut, kSize = 3, stride = 1, padding = 1)
        )
        self.bnrelu6 = BNPReLU(nOut)
    
    def forward(self, x):
        x2 = self.conv1x1_out(x) 
        x1 = self.conv_3x3(x2)
        max_out = self.max(x1)
        avg_out = self.avg(x1)

        # combined = torch.cat([avg_out, max_out], dim=1)

        output = self.bnrelu6(self.attn(max_out + avg_out) * x1)

        return output 


class FTModuleNoBN(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()

        self.conv_3x3 = nn.Sequential(
            ConvNoBn(nIn = nOut, nOut = nOut, kSize = 3, stride = 1, padding = 1),
            # nn.ReLU6() 
        )

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        # self.pa = PA(nOut)
        self.attn = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(nOut, nOut//2, 1),  # 输出通道调整为 mid_channels
            # nn.ReLU6(inplace= True),
            nn.Conv2d(nOut, nOut, 1),  # 输出通道调整为 mid_channels
            nn.Hardsigmoid(inplace= True)
        )

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(
                in_channels=nIn,
                out_channels=nOut,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU6()

            # ConvBn(nIn = nOut, nOut = nOut, kSize = 3, stride = 1, padding = 1)
        )
        # self.bnrelu6 = BNPReLU(nOut)
        self.relu6 = nn.ReLU6()
    
    def forward(self, x):
        x2 = self.conv1x1_out(x) 
        x1 = self.conv_3x3(x2)

        max_out = self.max(x1)
        avg_out = self.avg(x1)

        # combined = torch.cat([avg_out, max_out], dim=1)
        # combined = max_out + avg_out

        output = self.relu6(self.attn(max_out + avg_out) * x1)

        return output 


class DABModule(nn.Module):
    def __init__(self, nIn, d=2, kSize=3, dkSize=3):
        super().__init__()
        # self.mid_channels = (nIn * 3 + 1) // 2
        # self.mid_channels = nIn // 4
        self.mid_channels = nIn 
        # self.mid_channels = nIn
        # self.mid_channels = int(nIn // 1.5)
        # self.mid_channels += self.mid_channels % 2
        # self.mid_channels = nIn * 2

        # Branch A: one 3x3 depthwise separable convolution
        self.branch_a = nn.Sequential(
            nn.Conv2d(
                in_channels=nIn,
                out_channels=self.mid_channels,  
                # out_channels=nIn,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # BNPReLU(self.mid_channels)
            # ConvBn(nIn, self.mid_channels, 
            #     kSize=3, 
            #     stride=1, 
            #     padding=1),

        )


        self.branch_b1 = nn.Sequential(
            ConvBn(self.mid_channels, self.mid_channels, 
                kSize=3, 
                stride=1, 
                padding=1),
        )                       
                                
                                
        self.branch_b2 = nn.Sequential(
            ConvBn(self.mid_channels, self.mid_channels, 
                    kSize=5, 
                    stride=1, 
                    padding=2),
        )


        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=nIn,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BNPReLU(nIn)
        )
        
        # self.bn_relu6 = BNPReLU(nIn)
        # self.fusion = DynamicFusionGate(self.mid_channels)
        self.fusion = DABFusion(self.mid_channels)
        # self.eca = eca_layer(self.mid_channels)

    def forward(self, input):
        
        # Branch A
        branch_a_output = self.branch_a(input)
        # branch_b_output = self.branch_b1(branch_a_output) + branch_a_output
        # branch_c_output = self.branch_b2(branch_a_output) + branch_b_output
        # branch_a_output = input
        branch_b_output = self.branch_b1(branch_a_output) 
        branch_c_output = self.branch_b2(branch_a_output)
        # fusion = branch_b_output + branch_c_output
        fusion = self.fusion(branch_b_output, branch_c_output)

        att_output = self.conv1x1_out(fusion)
        # output = self.bn_relu6(att_output + input)
        output = att_output + input
        # output = att_output + input
        return output

class DABModule111(nn.Module):
    def __init__(self, nIn, d=2, kSize=3, dkSize=3):
        super().__init__()
        # self.mid_channels = (nIn * 3 + 1) // 2
        self.mid_channels = nIn // 2
        # self.mid_channels = nIn
        # self.mid_channels += self.mid_channels % 2
        # self.mid_channels = nIn * 2

        # Branch A: one 3x3 depthwise separable convolution
        self.branch_a = nn.Sequential(
            nn.Conv2d(
                in_channels=nIn,
                out_channels=self.mid_channels,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # BNPReLU(nIn)
        )
        
        self.branch_b = nn.Sequential(
            nn.Conv2d(
                in_channels=nIn,
                out_channels=self.mid_channels,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        
        self.branch_c = nn.Sequential(
            nn.Conv2d(
                in_channels=nIn,
                out_channels=self.mid_channels,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # BNPReLU(nIn)
        )

        self.attn = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Hardsigmoid(inplace= True)
        )

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels * 4,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BNPReLU(self.mid_channels * 4),
            nn.Conv2d(
                in_channels=self.mid_channels * 4,
                out_channels=nIn,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # BNPReLU(nIn)
        )


    def forward(self, input):
        
        # Branch A
        branch_a_output = self.branch_a(input)
        branch_b_output = self.branch_b(input) 
        branch_c_output = self.branch_c(input)

        fusion = self.attn(branch_a_output * branch_b_output) * branch_c_output


        att_output = self.conv1x1_out(fusion)
        # output = self.bn_relu6(att_output + input)
        output = att_output + input
        # output = att_output + input
        return output


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    
# class DownSamplingBlock(nn.Module):
#     def __init__(self, nIn, nOut):
#         super().__init__()

#         # 主路径：带下采样的卷积
#         self.conv_path = nn.Sequential(
#             ConvBn(nIn, nIn, kSize=3, stride=2, padding=1),
#             ConvBn(nIn, nOut, kSize=3, stride=1, padding=1),
#             # ConvBn(nOut, nOut, kSize=1, stride=2, padding=0),
#         )

#         self.conv1x1 = nn.Sequential(
#             ConvNoBn(nIn, nOut, kSize=1, stride=2, padding=0),
#         )
#         self.bn_relu6 = BNPReLU(nOut)
        
#     def forward(self, x):
#         conv_out = self.conv_path(x)
#         res = self.conv1x1(x)
#         return self.bn_relu6(conv_out + res)


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut, downsample_factor=2):
        super().__init__()
        assert downsample_factor >= 1, "downsample_factor must be ≥1"
        
        # 主路径参数计算
        kernel_size = downsample_factor * 2 - 1
        padding = downsample_factor // 2
        
        # 主下采样路径
        self.conv_path = nn.Sequential(
            
            ConvNoBn(nIn, nIn, 
                  kSize=kernel_size, 
                  stride=downsample_factor, 
                  padding=padding),
                #   BNPReLU(nIn),
            nn.Conv2d(
                in_channels=nIn,
                out_channels=nOut,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BNPReLU(nOut),
            # nn.BatchNorm2d(nOut),
            # nn.Sigmoid()
            # nn
        )

        # # 残差捷径路径
        self.conv1x1 = nn.Sequential(
            ConvNoBn(nIn, nOut, 
                    kSize=1, 
                    stride=downsample_factor, 
                    padding=0),
        )
        
        # self.bn_relu6 = BNPReLU(nOut)

    def forward(self, x):
        conv_out = self.conv_path(x)  # 主路径计算
        res = self.conv1x1(x)
        return conv_out + res
        # return conv_out + res
        # return self.bn_relu6(conv_out + res)  # 合并后激活

class UpsampleingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_factor=2):
        super(UpsampleingBlock, self).__init__()
        
        # 分解为多个2倍上采样步骤
        layers = []
        num_steps = int(math.log2(upsample_factor))
        if 2 ** num_steps != upsample_factor:
            raise ValueError(f"downsample_factor {upsample_factor} must be a power of 2.")
        
        for i in range(num_steps):
            # 添加转置卷积层
            layers.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0)
            )
            
            # 如果不是最后一次迭代，添加普通的 ConvBn 层
            if i < num_steps - 1:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=in_ch,  
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                        nn.BatchNorm2d(in_ch),
                        nn.Hardsigmoid(inplace= True)
                        # nn.Sigmoid()
                        # BNPReLU(in_ch)

                    )
                )
        
        # 在最后一次迭代后，添加一个特殊的 ConvBn 层，将通道数从 in_ch 转换为 out_ch
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,  
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(out_ch),
                nn.Hardsigmoid(inplace= True)
                # BNPReLU(out_ch)
            )
        )
        
        self.upsample_layers = nn.Sequential(*layers)  # 使用 *layers 将列表展开
    
    def forward(self, x):
        return self.upsample_layers(x)

# class UpsampleingBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
        
#         # self.conv3x3 = ConvBn(in_ch, out_ch*4, kSize=3, stride=1, padding=1)

        
#         # # Step 2: 使用 PixelShuffle 进一步优化结果
#         # self.pixel_shuffle = nn.PixelShuffle(2)
        
#         # # 可选：添加一个额外的卷积层用于平滑结果
#         # self.final_conv = ConvBn(out_ch, out_ch, kSize=3, stride=1, padding=1)

#         self.upconv1 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.conv3x3_1 = ConvBn(out_ch, out_ch, kSize=3, stride=1, padding=1)
#         # self.gate = GatedFusion(out_ch)

#     def forward(self, x):

#         # x = self.conv3x3(x_1)
        
#         # # Step 2: PixelShuffle 优化
#         # x = self.pixel_shuffle(x)
        
#         # # Step 3: 可选的最终卷积平滑
#         # x = self.final_conv(x)

#         x1 = self.upconv1(x)
#         x = self.conv3x3_1(x1)

#         return x

        
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y.squeeze(3).transpose(2, 1)).transpose(2, 1).unsqueeze(3)


        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize):
        super().__init__()

        self.conv3x3 = ConvBn(nIn, nOut, kSize=3, stride=1, padding= 1)
        self.ec1 = eca_layer(nOut)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=nOut,
                out_channels=nOut,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BNHardSwish(nOut)
        )
        
    def forward(self, input):

        output = self.conv3x3(input)
        output = self.ec1(output)
        output = self.conv1x1(output)

        return output

 
class TransferImgBin(nn.Module):
    def __init__(self, classes=1, blocks_num=[3, 4, 4, 4, 4, 4, 4, 4, 4, 3],  
                    img_size = [512, 512], layer_conv_num = [16, 32, 64, 128] ):
        super().__init__()

        # self.layer_conv_num = [16, 32, 64, 128, 256]
        self.layer_conv_num = layer_conv_num
        self.blocks_num = blocks_num

        self.init_conv = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=3,
        #         out_channels=self.layer_conv_num[0], 
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),
        #     BNPReLU(self.layer_conv_num[0]),

            # ConvNoBn(3, self.layer_conv_num[0], 3, 1, padding= 1),
            # ConvBn(3, self.layer_conv_num[0], 3, 1, padding=1),
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], 5, 1, padding=2),
            # FTModule(self.layer_conv_num[0], self.layer_conv_num[0]),
            # FTModuleNoBN(3, self.layer_conv_num[0]),
            ATModule(3, self.layer_conv_num[0])
            # PA(self.layer_conv_num[0]),
        )


        self.downsample_0 = DownSamplingBlock(self.layer_conv_num[0], self.layer_conv_num[0], downsample_factor=2)
        dilation_block_0 = [2, 3, 4, 5, 6, 8, 8, 8, 10, 10,10,10] 
        self.DAB_Block_0 = nn.Sequential(
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], 3, 1, padding=1),
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], 5, 1, padding=2),
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], 3, 1, padding=1),
            # FTModule(self.layer_conv_num[0], self.layer_conv_num[0]),
            ATModule(self.layer_conv_num[0], self.layer_conv_num[0])
            # FTModule(self.layer_conv_num[0], self.layer_conv_num[0]),
        )

        # self.DAB_Block_0 = nn.Sequential()
        # for i in range(0, self.blocks_num[0]):
        #     self.DAB_Block_0.add_module("DAB_Module_0_" + str(i), DABModule(self.layer_conv_num[0], d=dilation_block_0[i]))


        self.downsample_1 = DownSamplingBlock(self.layer_conv_num[0], self.layer_conv_num[1])
        dilation_block_1 = [2, 3, 4, 5, 6, 8, 8, 8, 10, 10,10,10] 
        # self.DAB_Block_1 = nn.Sequential(
        #     # ConvBn(self.layer_conv_num[1], self.layer_conv_num[1], 3, 1, padding=1),
        #     # ConvBn(self.layer_conv_num[1], self.layer_conv_num[1], 3, 1, padding=1),
        #     FTModule(self.layer_conv_num[1], self.layer_conv_num[1]),
        #     # FTModule(self.layer_conv_num[1], self.layer_conv_num[1]),
        # )
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, self.blocks_num[1]):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(self.layer_conv_num[1], d=dilation_block_1[i]))

        # DAB Block 2

        # dilation_block_2 = [2,2, 4, 4, 4, 4, 8, 8, 16, 16,32,32] 
        dilation_block_2 = [2,3, 4, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5] 
        self.downsample_2 = DownSamplingBlock(self.layer_conv_num[1], self.layer_conv_num[2], downsample_factor=2)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, self.blocks_num[2]):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(self.layer_conv_num[2], d=dilation_block_2[i]))


        self.downsample_3 = DownSamplingBlock(self.layer_conv_num[2], self.layer_conv_num[3])
        # dilation_block_3 = [2,2, 4, 4, 4, 4, 8, 8, 16, 16,32,32] 
        dilation_block_3 = [2,3, 4, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5] 
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, self.blocks_num[3]):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(self.layer_conv_num[3], dilation_block_3[i]))

        # self.downsample_4 = DownSamplingBlock(self.layer_conv_num[3], self.layer_conv_num[4])
        # # dilation_block_3 = [2,2, 4, 4, 4, 4, 8, 8, 16, 16,32,32] 
        # dilation_block_4 = [2,3, 4, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5] 
        # self.DAB_Block_4 = nn.Sequential()
        # for i in range(0, self.blocks_num[4]):
        #     self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
        #                                 DABModule(self.layer_conv_num[4], dilation_block_4[i]))


        # self.transformer1 = MS_Block(dim = self.layer_conv_num[3], num_heads = 8)
        # self.transformer2 = MS_Block(dim = self.layer_conv_num[3], num_heads = 8)
        # self.transformer2 = MS_Block(dim = 64, num_heads = 4)
         
        
#DECODER
        # dilation_block_4 = [2,3, 4, 5, 6, 6, 5, 5, 5, 5]
        # self.DAB_Block_4 = nn.Sequential()
        # for i in range(0, block_4):
        #    self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
        #                                DABModule(self.layer_conv_num[3], d = dilation_block_4[i]))

        # self.upsample_0 = UpsampleingBlock(self.layer_conv_num[4], self.layer_conv_num[3], upsample_factor=2)
        # self.DAB_Block_5 = nn.Sequential()
        # dilation_block_5 = [2,3, 4, 5, 6, 6,5, 5, 5, 5]
        # for i in range(0, self.blocks_num[5]):
        #     self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
        #                                 DABModule(self.layer_conv_num[3], d=dilation_block_5[i]))

        self.upsample_1 = UpsampleingBlock(self.layer_conv_num[3], self.layer_conv_num[2], upsample_factor=2)
        self.DAB_Block_6 = nn.Sequential()
        dilation_block_6 = [2,3, 4, 5, 6, 6,5, 5, 5, 5]
        for i in range(0, self.blocks_num[6]):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        DABModule(self.layer_conv_num[2], d=dilation_block_6[i]))



        self.upsample_2 = UpsampleingBlock(self.layer_conv_num[2], self.layer_conv_num[1], upsample_factor=2)
        dilation_block_7 = [2,3, 4, 5, 6, 6,5, 5, 5, 5, 5]
        self.DAB_Block_7 = nn.Sequential()
        for i in range(0, self.blocks_num[7]):
            self.DAB_Block_7.add_module("DAB_Module_7_" + str(i),
                                        DABModule(self.layer_conv_num[1], d = dilation_block_7[i]))

        self.upsample_3 = UpsampleingBlock(self.layer_conv_num[1], self.layer_conv_num[0])
        dilation_block_8 = [2,3, 4, 5, 5, 5,5, 5, 5, 5,5]
        self.DAB_Block_8 = nn.Sequential(
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], kSize=3, stride=1, padding=1),
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], kSize=3, stride=1, padding=1),
            # FTModule(self.layer_conv_num[0], self.layer_conv_num[0]),
            ATModule(self.layer_conv_num[0], self.layer_conv_num[0]),
            ATModule(self.layer_conv_num[0], self.layer_conv_num[0])
        )
        # self.DAB_Block_8 = nn.Sequential()
        # for i in range(0, self.blocks_num[7]):
        #     self.DAB_Block_8.add_module("DAB_Module_8_" + str(i),
        #                                 DABModule(self.layer_conv_num[0], d=dilation_block_8[i]))

        self.upsample_4 = UpsampleingBlock(self.layer_conv_num[0], self.layer_conv_num[0])
        dilation_block_9 = [2,3, 4, 5, 5, 5,5, 5, 5]
        self.DAB_Block_9 = nn.Sequential(
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], kSize=3, stride=1, padding=1),
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], kSize=3, stride=1, padding=1),
            ATModule(self.layer_conv_num[0], self.layer_conv_num[0])

            # FTModule(self.layer_conv_num[0], self.layer_conv_num[0]),
        )
        # self.DAB_Block_9 = nn.Sequential()
        # for i in range(0, self.blocks_num[8]):
        #     self.DAB_Block_9.add_module("DAB_Module_9_" + str(i),
        #                                 DABModule(self.layer_conv_num[0], d=dilation_block_9[i]))
 
        # self.gate_fusion1 = GatedFusion(in_channels= self.layer_conv_num[3]) 
        # self.gate_fusion2 = GatedFusion(in_channels= self.layer_conv_num[2]) 
        # self.gate_fusion3 = GatedFusion(in_channels= self.layer_conv_num[1]) 
        # self.gate_fusion4 = GatedFusion(in_channels= self.layer_conv_num[0]) 

        self.gate_fusion1 = GatedFusion(in_channels= self.layer_conv_num[2]) 
        self.gate_fusion2 = GatedFusion(in_channels= self.layer_conv_num[1]) 
        self.gate_fusion3 = GatedFusion(in_channels= self.layer_conv_num[0]) 
        self.gate_fusion4 = GatedFusion(in_channels= self.layer_conv_num[0]) 

        
        self.classifier = nn.Sequential(
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0]//2, kSize=3, stride=1, padding=1),
            # ConvBn(self.layer_conv_num[0]//2, self.layer_conv_num[0]//4, kSize=3, stride=1, padding=1),
            # ConvBn(self.layer_conv_num[0]//4, 3, kSize=3, stride=1, padding=1),
            # PA(self.layer_conv_num[0]//4),
            # ConvBn(self.layer_conv_num[0], self.layer_conv_num[0], 3, 1, padding=1),

            # PA(self.layer_conv_num[0]),
            ConvNoBn(self.layer_conv_num[0], self.layer_conv_num[0], kSize=3, stride=1, padding=1),
            # nn.Hardsigmoid(),
            nn.Sigmoid(),

            nn.Conv2d(
                in_channels=self.layer_conv_num[0],
                # in_channels=3,
                out_channels=classes,  
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid()
        )


    def forward(self, input):

        output0_00 = self.init_conv(input)
        output0_0 = self.downsample_0(output0_00)
        output0 = self.DAB_Block_0(output0_0)

        # # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        # print(f"output1:{output1.shape}")

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        # print(f"output2:{output2.shape}")

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        # print(f"output3:{output3.shape}")

#Transformer

        # output4_0 = self.transformer1(self.gate_fusion0(output3, output3_0))
        # output4_0 = self.transformer1(output3)
        # output4_0 = self.DAB_Block_3_1(output3)
        # print(f"output4_0:{output4.shape}")
        # output3_1 = self.downsample_4(output3)
        # output3_2 = self.DAB_Block_4(output3_1)

    

#DECODER            
        # output4_1 = self.DAB_Block_4(self.gate_fusion1_1(output4_0,output3))
        # output4_1 = self.DAB_Block_4(output4_0)

        # output4_00 = self.upsample_0(output3)
        # output4_0 = self.DAB_Block_5(output4_00)

        # output4 = self.upsample_1(output4_0)
        output4 = self.upsample_1(output3)
        # output4 = self.upsample_1(output4_0)
        output5_0 = self.DAB_Block_6(self.gate_fusion1(output4, output2))
        
        # output5 = self.upsample_2(self.gate_fusion2(output5_0, output4))
        output5 = self.upsample_2(output5_0)
        output6_0 = self.DAB_Block_7(self.gate_fusion2(output5, output1))

        # output6 = self.upsample_3(self.gate_fusion3(output6_0, output5))
        output6 = self.upsample_3(output6_0)
        # output7_0 = self.DAB_Block_8(self.gate_fusion4_1(output6, output0))
        output7_0 = self.DAB_Block_8(self.gate_fusion3(output6, output0))

        # output7 = self.upsample_4(self.gate_fusion4(output7_0, output6))
        output7 = self.upsample_4(output7_0)
        # output7 = self.upsample_4(self.gate_fusion4(output6, output0_0) + output7_0)
        # output7 = self.upsample_4(output7_0)
        # output8 = self.DAB_Block_9(output7 + output0_00)
        output8 = self.DAB_Block_9(self.gate_fusion4(output7, output0_00))
        # output8 = self.DAB_Block_9(output7)
        # output8 = self.DAB_Block_9(output7 )

        # out = self.classifier(self.gate_fusion5(output8, output0_0))
        out = self.classifier(output8)
        return out

def create_model(num_class = 1,
                num_block = [3, 4, 8, 2, 2, 2],
                img_size = [256, 256],
                layer_conv_num = [16, 32, 64, 128],
                **kwargs):

    model = TransferImgBin(num_class, 
                            blocks_num= num_block,
                           img_size = [256, 256],
                           layer_conv_num = layer_conv_num,
                            **kwargs)

    # model = UNet(n_channels=3, n_classes=num_class, **kwargs)

    return model

