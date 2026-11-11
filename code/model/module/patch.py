import os
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F


def normalize(x):
    return x.mul_(2).add_(-1)


# def same_padding(images, ksizes, strides, rates):
#     # assert len(images.size()) == 4
#     batch_size, channel, rows, cols = images.size()
#     out_rows = (rows + strides[0] - 1) // strides[0]  ##rows
#     out_cols = (cols + strides[1] - 1) // strides[1]  ##cols
#     effective_k_row = (ksizes[0] - 1) * rates[0] + 1  ##3
#     effective_k_col = (ksizes[1] - 1) * rates[1] + 1  ##3
#     padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)  ##2
#     padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)  ##2
#     # Pad the input
#     padding_top = int(padding_rows / 2.)  ##1
#     padding_left = int(padding_cols / 2.)  ##1
#     padding_bottom = padding_rows - padding_top  ##1
#     padding_right = padding_cols - padding_left  ##1
#     paddings = (padding_left, padding_right, padding_top, padding_bottom)
#     # images = torch.nn.ZeroPad2d(paddings)(images)
#     images = F.pad(images, paddings, "constant", 0.0)  # 使用 F.pad 函数

#     return images


# # def extract_image_patches(images, ksizes, strides, rates, padding='same'):
# def extract_image_patches(images, ksizes, strides, rates):
#     """
#     Extract patches from images and put them in the C output dimension.
#     :param padding:
#     :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
#     :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
#      each dimension of images
#     :param strides: [stride_rows, stride_cols]
#     :param rates: [dilation_rows, dilation_cols]
#     :return: A Tensor
#     """
#     # assert len(images.size()) == 4
#     # assert padding in ['same', 'valid']
#     # batch_size, channel, height, width = images.size()

#     # if padding == 'same':
#     #     images = same_padding(images, ksizes, strides, rates)
#     # elif padding == 'valid':
#     #     pass
#     # else:
#     #     raise NotImplementedError('Unsupported padding type: {}.\
#     #             Only "same" or "valid" are supported.'.format(padding))

#     images = same_padding(images, ksizes, strides, rates)

#     unfold = torch.nn.Unfold(kernel_size=ksizes,
#                              dilation=rates,
#                              padding=0,
#                              stride=strides)
#     patches = unfold(images)
#     return patches  # [N, C*k*k, L], L is the total number of such blocks


def same_padding(images, ksizes, strides, rates):
    """
    
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: Padded images tensor
    """
    batch_size, channel, height, width = images.size()
    effective_ksize_rows = (ksizes[0] - 1) * rates[0] + 1
    effective_ksize_cols = (ksizes[1] - 1) * rates[1] + 1

    padding_rows = max(0, (height * strides[0] - height + effective_ksize_rows - strides[0]) // 2)
    padding_cols = max(0, (width * strides[1] - width + effective_ksize_cols - strides[1]) // 2)

    paddings = (padding_cols, padding_cols, padding_rows, padding_rows)
    images = F.pad(images, paddings, "constant", 0.0)  # 使用 F.pad 函数
    return images

class CustomUnfold(torch.nn.Module):
    def __init__(self, kernel_size_rows, kernel_size_cols, dilation_rows=1, dilation_cols=1, padding=0, stride_rows=1, stride_cols=1):
        super().__init__()
        self.kernel_size_rows = kernel_size_rows
        self.kernel_size_cols = kernel_size_cols
        self.dilation_rows = dilation_rows
        self.dilation_cols = dilation_cols
        self.padding = padding
        self.stride_rows = stride_rows
        self.stride_cols = stride_cols

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 计算输出的 patch 数量
        out_height = (height - self.kernel_size_rows + 2 * self.padding) // self.stride_rows + 1
        out_width = (width - self.kernel_size_cols + 2 * self.padding) // self.stride_cols + 1
        num_patches = out_height * out_width

        # 初始化一个空列表来存储所有 patches
        patches_list = []

        # 遍历每个位置，提取 patches
        for i in range(out_height):
            for j in range(out_width):
                start_row = i * self.stride_rows - self.padding
                start_col = j * self.stride_cols - self.padding
                end_row = start_row + self.kernel_size_rows * self.dilation_rows
                end_col = start_col + self.kernel_size_cols * self.dilation_cols

                # 提取当前 patch
                patch = x[:, :, start_row:end_row:self.dilation_rows, start_col:end_col:self.dilation_cols]
                patches_list.append(patch)

        # 将 patches 列表转换为张量
        patches_tensor = torch.stack(patches_list, dim=-1)  # [N, C, ksize_rows, ksize_cols, L]
        patches_tensor = patches_tensor.view(batch_size, channels * self.kernel_size_rows * self.kernel_size_cols, num_patches)

        return patches_tensor  # [N, C*k*k, L]

def extract_image_patches(images, ksizes, strides, rates):
    """
    从图像中提取 patches 并将它们放在输出维度 C 中。
    
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor of shape [N, C*k*k, L], where L is the total number of such blocks
    """

    images = same_padding(images, ksizes, strides, rates)

    # # 使用自定义的 Unfold 类来避免 TorchScript 编译问题
    # unfold = CustomUnfold(
    #     kernel_size_rows=ksizes[0],
    #     kernel_size_cols=ksizes[1],
    #     dilation_rows=rates[0],
    #     dilation_cols=rates[1],
    #     padding=0,
    #     stride_rows=strides[0],
    #     stride_cols=strides[1]
    # )
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)

    patches = unfold(images)
    return patches  # [N, C*k*k, L]


# class ExtractImagePatches(torch.nn.Module):
#     def __init__(self, ksizes, strides, rates, padding=0):
#         """
#         从图像中提取 patches 并将它们放在输出维度 C 中。
        
#         :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for each dimension of images
#         :param strides: [stride_rows, stride_cols]
#         :param rates: [dilation_rows, dilation_cols]
#         :param padding: Padding value (default is 0)
#         """
#         super(ExtractImagePatches, self).__init__()
#         self.ksizes = ksizes
#         self.strides = strides
#         self.rates = rates
#         self.padding = padding

#         self.unfold = torch.nn.Unfold(kernel_size=ksizes,
#                              dilation=rates,
#                              padding=0,
#                              stride=strides)

#     def same_padding(self, images):
#         """
#         计算并应用 'same' 填充。
        
#         :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
#         :return: Padded images tensor
#         """
#         # batch_size, channel, height, width = images.size()
#         # effective_ksize_rows = (self.ksizes[0] - 1) * self.rates[0] + 1
#         # effective_ksize_cols = (self.ksizes[1] - 1) * self.rates[1] + 1

#         # padding_rows = max(0, (height * self.strides[0] - height + effective_ksize_rows - self.strides[0]) // 2)
#         # padding_cols = max(0, (width * self.strides[1] - width + effective_ksize_cols - self.strides[1]) // 2)

#         # paddings = (padding_cols, padding_cols, padding_rows, padding_rows)
#         # print(f"{paddings}")
#         # images = F.pad(images, paddings, "constant", 0.0)  # 使用 F.pad 函数
#         images = F.pad(images, (1,1,1,1), "constant", 0.0)  # 使用 F.pad 函数
#         return images

#     # def same_padding(self, images):
#     #     """
#     #     计算并应用 'same' 填充。
        
#     #     :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
#     #     :return: Padded images tensor
#     #     """
#     #     batch_size, channel, height, width = images.size()
        
#     #     # 计算有效的卷积核大小
#     #     effective_ksize_rows = (self.ksizes[0] - 1) * self.rates[0] + 1
#     #     effective_ksize_cols = (self.ksizes[1] - 1) * self.rates[1] + 1

#     #     # 计算填充大小
#     #     padding_rows = torch.clamp(
#     #         (height * self.strides[0] - height + effective_ksize_rows - self.strides[0]) // 2,
#     #         min=0
#     #     )
#     #     padding_cols = torch.clamp(
#     #         (width * self.strides[1] - width + effective_ksize_cols - self.strides[1]) // 2,
#     #         min=0
#     #     )

#     #     # 确保填充大小是整数
#     #     padding_rows = padding_rows.to(torch.int64)
#     #     padding_cols = padding_cols.to(torch.int64)

#     #     # 构建填充参数
#     #     paddings = (padding_cols, padding_cols, padding_rows, padding_rows)
#     #     print(f"{paddings}")

#     #     # 应用填充
#     #     images = F.pad(images, paddings, "constant", 0.0)

#     #     return images

#     def forward(self, images):
#         """
#         前向传播函数，执行图像填充和 patch 提取。
        
#         :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
#         :return: A Tensor of shape [N, C*k*k, L], where L is the total number of such blocks
#         """
#         # 应用 'same' 填充
#         padded_images = self.same_padding(images)

#         # 使用自定义的 Unfold 模块提取 patches
#         patches = self.unfold(padded_images)
#         return patches  # [N, C*k*k, L]

class ExtractImagePatches(torch.nn.Module):
    def __init__(self, ksizes, strides, rates, padding = 0):
        """
        从图像中提取 patches 并将它们放在输出维度 C 中。
        
        :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for each dimension of images
        :param strides: [stride_rows, stride_cols]
        :param rates: [dilation_rows, dilation_cols]
        :param padding: Padding mode, can be 'valid' or 'same'
        """
        super(ExtractImagePatches, self).__init__()
        self.ksizes = ksizes
        self.strides = strides
        self.rates = rates

    def same_padding(self, images):
        """
        计算并应用 'same' 填充。
        
        :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
        :return: Padded images tensor
        """
        # print(f"image_size:{images.size()}")
        batch_size, channels, height, width = images.size()
        effective_ksize_rows = (self.ksizes[0] - 1) * self.rates[0] + 1
        effective_ksize_cols = (self.ksizes[1] - 1) * self.rates[1] + 1

        padding_rows = max(0, (height * self.strides[0] - height + effective_ksize_rows - self.strides[0]) // 2)
        padding_cols = max(0, (width * self.strides[1] - width + effective_ksize_cols - self.strides[1]) // 2)

        paddings = (padding_cols, padding_cols, padding_rows, padding_rows)
        images = F.pad(images, paddings, "constant", 0.0)  # 使用 F.pad 函数

        return images


class ExtractImagePatches(torch.nn.Module):
    def __init__(self, ksizes, strides, rates, padding = 0):
        """
        从图像中提取 patches 并将它们放在输出维度 C 中。
        
        :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for each dimension of images
        :param strides: [stride_rows, stride_cols]
        :param rates: [dilation_rows, dilation_cols]
        :param padding: Padding mode, can be 'valid' or 'same'
        """
        super(ExtractImagePatches, self).__init__()
        self.ksizes = ksizes
        self.strides = strides
        self.rates = rates

    def same_padding(self, images):
        """
        计算并应用 'same' 填充。
        
        :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
        :return: Padded images tensor
        """
        # print(f"image_size:{images.size()}")
        batch_size, channels, height, width = images.size()
        effective_ksize_rows = (self.ksizes[0] - 1) * self.rates[0] + 1
        effective_ksize_cols = (self.ksizes[1] - 1) * self.rates[1] + 1

        padding_rows = max(0, (height * self.strides[0] - height + effective_ksize_rows - self.strides[0]) // 2)
        padding_cols = max(0, (width * self.strides[1] - width + effective_ksize_cols - self.strides[1]) // 2)

        paddings = (padding_cols, padding_cols, padding_rows, padding_rows)
        images = F.pad(images, paddings, "constant", 0.0)  # 使用 F.pad 函数

        return images

    def forward(self, images):
        """
        前向传播函数，执行图像填充和 patch 提取。
        
        :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
        :return: A Tensor of shape [N, C*k*k, L], where L is the total number of such blocks
        """
        # if self.padding_mode == 'same':
        images = self.same_padding(images)

        batch_size, channels, height, width = images.size()

        # Calculate output dimensions after applying the kernel with given stride and rate
        out_height = (height - (self.ksizes[0] - 1) * self.rates[0] - 1) // self.strides[0] + 1
        out_width = (width - (self.ksizes[1] - 1) * self.rates[1] - 1) // self.strides[1] + 1

        # Unfold the images manually using indexing
        patches = []
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]
                h_end = h_start + self.ksizes[0] * self.rates[0]
                w_end = w_start + self.ksizes[1] * self.rates[1]

                # Extract the patch
                patch = images[:, :, h_start:h_end:self.rates[0], w_start:w_end:self.rates[1]]
                patches.append(patch)

        # Stack all patches into a single tensor
        patches = torch.stack(patches, dim=-1)

        # Rearrange the patches into the desired output format [N, C*k*k, L]
        patches = patches.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, -1, out_height * out_width)

        return patches  # [N, C*k*k, L]

def reverse_patches(images, out_size, ksizes, strides, padding = 0):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    unfold = torch.nn.Fold(output_size=out_size,
                           kernel_size=ksizes,
                           dilation=1,
                           padding=padding,
                           stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


# class ReversePatches(torch.nn.Module):
#     def __init__(self, out_size, ksizes, strides, padding = 1):
#         super(ReversePatches, self).__init__()
#         self.unfold = torch.nn.Fold(output_size=out_size,
#                             kernel_size=ksizes,
#                             dilation=1,
#                             padding=padding,
#                             stride=strides)

#         self.out_size = out_size
#         self.strides = strides
#         self.padding = padding
#         self.kernel_size = ksizes
        
#     def forward(self, img):
#         patches = self.unfold(img)

#         return patches


class ReversePatches(torch.nn.Module):
    def __init__(self, out_size, ksizes, strides, padding=1):
        """
        将 patches 重新组合成原始图像。

        :param out_size: (out_height, out_width). 输出图像的尺寸
        :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for each dimension of images
        :param strides: [stride_rows, stride_cols]
        :param padding: Padding value (default is 1)
        """
        super(ReversePatches, self).__init__()
        self.out_size = out_size
        self.ksizes = ksizes
        self.strides = strides
        self.padding = padding

        self.conv3x3 = torch.nn.Conv2d(288, 288//(ksizes[0]*ksizes[1]), kernel_size=ksizes,
                              stride=strides, padding=padding)



    def forward(self, img):
        batch_size, num_patches, c_kh_kw = img.size()

        feature_map = img.reshape(batch_size, num_patches, self.out_size[1], self.out_size[0])
        feature_map = self.conv3x3(feature_map)

        return feature_map

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x