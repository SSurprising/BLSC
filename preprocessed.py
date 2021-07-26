import os 
import shutil

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

file_path = '/home/zjm/Data/HaN_OAR/'
new_file_path = '/home/zjm/Data/HaN_OAR_256/'

if not os.path.exists(new_file_path):
    os.makedirs(new_file_path)

HU_lower = -1000
HU_upper = 1000

slice_thickness = 3
down_scale = 0.5

for root, dirs, files in os.walk(file_path):
    if 'image.nii' in files:
        print(root, dirs, files)
        subset = root.split('/')[-2]
        patient = root.split('/')[-1]

        new_img_path = os.path.join(new_file_path, subset, patient)
        if not os.path.exists(new_img_path):
            os.makedirs(new_img_path)

        # print(new_img_path)

        img   = sitk.ReadImage(os.path.join(root, 'image.nii'))
        label = sitk.ReadImage(os.path.join(root, 'label.nii'))

        img_arr   = sitk.GetArrayFromImage(img)
        label_arr = sitk.GetArrayFromImage(label)

        new_img_arr   = ndimage.zoom(img_arr, ((img.GetSpacing()[-1] / slice_thickness), down_scale, down_scale), order=3)
        new_label_arr = ndimage.zoom(label_arr, ((label.GetSpacing()[-1] / slice_thickness), down_scale, down_scale), order=0)

        new_img_arr[new_img_arr > HU_upper] = HU_upper
        new_label_arr[new_label_arr < HU_lower] = HU_lower

        new_img   = sitk.GetImageFromArray(new_img_arr)
        new_label = sitk.GetImageFromArray(new_label_arr)

        sitk.WriteImage(new_img, os.path.join(new_img_path, 'image.nii'))
        sitk.WriteImage(new_label, os.path.join(new_img_path, 'label.nii'))



