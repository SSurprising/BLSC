#encoding=UTF-8
"""
随机取样方式下的数据集
"""

import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import numpy as np

class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, slice_size, slice_expand=0):

        self.size = slice_size
        self.slice_expand = slice_expand

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = os.listdir(seg_dir)
        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([B, 1, 48, 256, 256]) torch.Size([B, 48, 256, 256])
        """

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        idx = np.where(seg_array != 0)
        idx_min = min(idx[0])
        idx_max = max(idx[0])

        idx_min = max(0, idx_min - self.slice_expand)
        idx_max = min(idx_max + self.slice_expand, seg_array.shape[0]-1)

        # 在slice平面内随机选取48张slice
        start_slice = random.randint(idx_min, idx_max - self.size)
        end_slice = start_slice + self.size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)
