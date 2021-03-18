#encoding=UTF-8

import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import numpy as np


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, slice_size, cts, gts, slice_expand=0):

        self.size = slice_size
        self.slice_expand = slice_expand
        self.random_min = int(0.5 * slice_size)
        self.cts = cts
        self.gts = gts

    def __getitem__(self, index):


        ct_array = self.cts[index]
        seg_array = self.gts[index]

        idx = np.where(seg_array != 0)
        idx_min = min(idx[0])
        idx_max = max(idx[0])

        idx_min = max(0, idx_min - self.slice_expand)
        idx_max = min(idx_max + self.slice_expand, seg_array.shape[0]-1)

        start_slice = random.randint(idx_min, idx_min + 12)


        ct_array_list  = []
        seg_array_list = []
        while (start_slice + self.size) < idx_max + 1:
            end_slice   = start_slice + self.size
            ct_array_temp  = ct_array[start_slice:end_slice, :, :]
            seg_array_temp = seg_array[start_slice:end_slice, :, :]
            ct_array_list.append(torch.FloatTensor(ct_array_temp).unsqueeze(0))
            seg_array_list.append(torch.FloatTensor(seg_array_temp))

            start_slice = random.randint(start_slice + self.random_min, end_slice)

        end_slice = random.randint(idx_max - 12, idx_max)
        start_slice = end_slice - self.size
        ct_array_temp  = ct_array[start_slice:end_slice, :, :]
        seg_array_temp = seg_array[start_slice:end_slice, :, :]
        ct_array_list.append(torch.FloatTensor(ct_array_temp).unsqueeze(0))
        seg_array_list.append(torch.FloatTensor(seg_array_temp))



        return ct_array_list, seg_array_list

    def __len__(self):

        return len(self.cts)