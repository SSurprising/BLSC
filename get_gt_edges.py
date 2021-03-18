import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import skimage
from skimage import io, color, feature, measure
import matplotlib.pyplot as plt
import copy
import json
import argparse
from time import time

organ_list = [
    'esophagus',
    'heart',
    'trachea',
    'aorta'
]

# visualizing the pred

print('====================================================')
print('Getting the coordinates of edges from the ground truth: ')
print('====================================================')

label_path = '/openbayes/input/input0/val/GT_raw/Patient_01.nii'
save_path = '/output/coordinates/gt/'



if not os.path.exists(label_path):
    print('No such path： ', label_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for i in range(1, 41):
    start_time = time()
    if i > 10:
        mode = 'train'
    else:
        mode = 'val'
        
    label_path = '/openbayes/input/input0/' + mode + '/GT_raw/Patient_01.nii'
    
    if i < 10:
        new_id = '0' + str(i)
    else:
        new_id = str(i)
    
    patient_name = 'Patient_' + new_id
    patient_temp = {'name': patient_name}
    edges_list = []
    
    print(patient_name)
    label_path_temp = label_path.replace('_01', '_' + new_id)

    pred = sitk.ReadImage(label_path_temp)
    pred = sitk.GetArrayFromImage(pred)
#     pred = pred[::-1]
    
    patient_temp['slice_number'] = pred.shape[0]
    
    for slice in range(pred.shape[0]):
        
        slice_temp = {'esophagus':[-1], 'heart':[-1], 'trachea':[-1], 'aorta':[-1]}
        
        label      = pred[slice]
        label_list = np.unique(label)
        
        # print('slice_index:', slice)
        # print('label_list:', label_list)
        
        for label_id in label_list:
            if label_id != 0:
                label_temp = copy.deepcopy(label)
                label_temp[label_temp != label_id] = 0

                label_temp[label_temp == label_id] = 1
                
                organ_label_coordinates = []
                
                labels = measure.label(label_temp)
#                 print('pixels = ', labels.sum())
                # regionprops中标签为0的区域会被自动忽略
                for region in measure.regionprops(labels):
                    region_image = np.zeros(label_temp.shape)
                    
                    
                    # region_image[region.coords] = 1
                    
                    region_image[region.coords[:, 0], region.coords[:, 1]] = 1
                                     
                    edges = feature.canny(region_image, low_threshold=1, high_threshold=1)
                    index = np.where(edges)
                    
                    
                    # 两个列表交叉
                    index_list = [int(index[i%2][i//2]) for i in range (len(index[0]) + len(index[1]))]
                    
                    organ_label_coordinates.append(index_list)
                
                slice_temp[organ_list[label_id-1]] = organ_label_coordinates
                
        patient_temp[str(slice)] = slice_temp
    edges_list.append(patient_temp)
    
    with open(save_path + patient_name + '.json', 'w') as f_obj:
        json.dump(edges_list, f_obj)
    print('time = ', time() - start_time)
    print('--------------------------------------------------\n')


