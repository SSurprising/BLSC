import os 
import shutil

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

ct_path = '/openbayes/input/input0/train/CT_raw/'
gt_path = ct_path.replace('CT', 'GT')

new_ct_path = '/output/processed_dataset/train/CT/'
new_gt_path = new_ct_path.replace('CT', 'GT')

if not os.path.exists(new_ct_path):
    os.makedirs(new_ct_path)
if not os.path.exists(new_gt_path):
    os.makedirs(new_gt_path)


HU_lower = -1000
HU_upper = 1000

slice_thickness = 2.5
down_scale = 0.5

ct_list = os.listdir(ct_path)

for ct_name in ct_list:
    print(ct_name)
    
    ct = sitk.ReadImage(ct_path + ct_name)
    gt = sitk.ReadImage(gt_path + ct_name)

    ct_array = sitk.GetArrayFromImage(ct)
    gt_array = sitk.GetArrayFromImage(gt)
    
    new_ct_array = ndimage.zoom(ct_array, ((ct.GetSpacing()[-1] / slice_thickness), down_scale, down_scale), order=3)
    new_gt_array = ndimage.zoom(gt_array, ((gt.GetSpacing()[-1] / slice_thickness), down_scale, down_scale), order=0)
    
    new_ct_array[new_ct_array > HU_upper] = HU_upper
    new_ct_array[new_ct_array < HU_lower] = HU_lower
    
    new_ct = sitk.GetImageFromArray(new_ct_array)
    new_gt = sitk.GetImageFromArray(new_gt_array)
    
    sitk.WriteImage(new_ct, new_ct_path + ct_name)
    sitk.WriteImage(new_gt, new_gt_path + ct_name)
    
    

