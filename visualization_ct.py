import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import skimage
from skimage import io, color, feature
import copy

print('====================================================')
print('visualizing the raw images')
print('====================================================')



HU = 500

for i in range(1, 2):
    
    ct_path = '/openbayes/input/input0/processed/CT/Patient_01.nii'
    gray_save_path = '/output/visualization/CT/Patient_01/'
    gray_window_save_path = '/output/visualization/CT/Patient_01_window/'
    if not os.path.exists(gray_save_path):
        os.makedirs(gray_save_path)
    if not os.path.exists(gray_window_save_path):
        os.makedirs(gray_window_save_path)
    
    ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)


    for index in range(ct_array.shape[0]):
        ct_slice = ct_array[index]

        # write gray image
        gray_path = gray_save_path + '/' + str(index) + '.png'
        cv.imwrite(gray_path, ct_slice)
        
        ct_slice[ct_slice > 350] = 350
        ct_slice[ct_slice < -350] = -350
        
        ct_slice = (ct_slice + 350) / 700 * 255
        gray_window_path = gray_window_save_path + '/' + str(index) + '.png'
        cv.imwrite(gray_window_path, ct_slice)
        

print('--------------------------------------------------\n')

    