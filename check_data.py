import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter

# data_dir = '/home/zjm/Data/HaN_OAR/'
data_dir = '/home/zjm/Data/HaN_OAR_256/'

spacing_couter_flag = False
organ_slice_number = []
if spacing_couter_flag:
    thickness = []
    spacing = []

for subset in os.listdir(data_dir):
    print('\nDataset:', subset)
    for patient in os.listdir(data_dir + subset):
        image = sitk.ReadImage(data_dir + subset + '/' + patient + '/image.nii')
        label = sitk.ReadImage(data_dir + subset + '/' + patient + '/label.nii')

        img_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)

        # print('patien:', patient)
        # print('image size:', img_arr.shape)
        # print('labels:', np.unique(label_arr))

        idx = np.where(label_arr != 0)
        idx_min = min(idx[0])
        idx_max = max(idx[0])
        organ_slice_number.append(idx_max - idx_min + 1)

        # print('imgagee spaceing:', image.GetSpacing())
        # print('--------------------------------\n')

        if spacing_couter_flag:
            thickness.append(image.GetSpacing()[-1])
            spacing.append(image.GetSpacing()[:-1])
if spacing_couter_flag:
    print('thickness:', Counter(thickness))
    print('spacing:', Counter(spacing))

# the number of slices for a patient varies from 82 to 120
print(organ_slice_number)