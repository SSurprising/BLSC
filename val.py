import os
from time import time

import torch
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk
import xlsxwriter as xw
import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _net.DenseUnet4_EtoE_3d4 import Net

print('==========testing==========')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


current_dir = os.path.dirname(os.path.abspath(__file__))
val_ct_dir  = '/openbayes/input/input0/val/CT_raw/'
val_seg_dir = val_ct_dir.replace('CT', 'GT')

organ_pred_dir = '/output/pred_result/'
if not os.path.exists(organ_pred_dir):
    os.makedirs(organ_pred_dir)

slice_size = 48

module_dir = os.path.join(current_dir, 'checkpoints') 
if not os.path.exists(module_dir + '/best.pth'):
    print('No such file:', module_dir + '.', 'We will use our standard model.')
    module_dir += '/standard_best.pth'
else:
    module_dir += '/best.pth'

upper = 1000
lower = -upper
down_scale = 0.5
size = 48
slice_thickness = 2.5

organ_list = [
    'esophagus',
    'heart',
    'trachea',
    'aorta',
]

dice_average = [0, 0, 0, 0]

# 创建一个表格对象，并添加一个sheet，后期配合window的excel来出图
excel_path = '/output/dice_score/'
if not os.path.exists(excel_path):
    os.makedirs(excel_path)
    
workbook = xw.Workbook(excel_path + 'dice_score.xlsx')
worksheet = workbook.add_worksheet('result')

# 设置单元格格式
bold = workbook.add_format()
bold.set_bold()

center = workbook.add_format()
center.set_align('center')

center_bold = workbook.add_format()
center_bold.set_bold()
center_bold.set_align('center')

worksheet.set_column(1, len(os.listdir(val_ct_dir)), width=15)
worksheet.set_column(0, 0, width=30, cell_format=center_bold)
worksheet.set_row(0, 20, center_bold)

# 写入文件名称
worksheet.write(0, 0, 'file name')
for index, file_name in enumerate(os.listdir(val_ct_dir), start=1):
    worksheet.write(0, index, file_name)
worksheet.write(0, len(os.listdir(val_ct_dir))+1, 'mean')
    
# 写入各项评价指标名称
for index, organ_name in enumerate(organ_list, start=1):
    worksheet.write(index, 0, organ_name)
worksheet.write(5, 0, 'speed')
worksheet.write(6, 0, 'shape')


# 定义网络并加载参数
net = torch.nn.DataParallel(Net(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()


# 开始正式进行测试
for file_index, file in enumerate(os.listdir(val_ct_dir)):

    print(file)
    start_time = time()

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(val_ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    
    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    
    # 在轴向上进行切块取样
    flag = False
    start_slice = 0
    end_slice = start_slice + size - 1
    ct_array_list = []

    while end_slice <= ct_array.shape[0] - 1:
        ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

        start_slice = end_slice + 1
        end_slice = start_slice + size - 1

    # 当无法整除的时候反向取最后一个block
    if end_slice is not ct_array.shape[0] - 1:
        flag = True
        count = ct_array.shape[0] - start_slice
        ct_array_list.append(ct_array[-size:, :, :])

    outputs_list = []
    with torch.no_grad():
        for ct_array in ct_array_list:

            ct_tensor = torch.FloatTensor(ct_array).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)

            ct_tensor /= 1000

            outputs = net(ct_tensor)
            outputs = outputs.squeeze()

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            outputs_list.append(outputs.cpu().detach().numpy())
            del outputs

    # 执行完之后开始拼接结果
    pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
    if flag is False:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('img', 'label')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0)
    print(pred_seg.shape);print(seg_array.shape)
    pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    print(pred_seg.shape)
    pred_seg = np.argmax(pred_seg, axis=0)
    pred_seg = np.round(pred_seg).astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)

    worksheet.write(6, file_index + 1, pred_seg.shape[0])

    # 计算每一种器官的dice系数，并将结果写入表格中存储
    for organ_index, organ in enumerate(organ_list, start=1):

        pred_organ = np.zeros(pred_seg.shape)
        target_organ = np.zeros(seg_array.shape)

        pred_organ[pred_seg == organ_index] = 1
        target_organ[seg_array == organ_index] = 1

        # 如果该例数据中不存在某一种器官，在表格中记录 None 跳过即可
        if target_organ.sum() == 0:
            worksheet.write(organ_index, file_index + 1, 'None')
        else:
            dice = (2 * pred_organ * target_organ).sum() / (pred_organ.sum() + target_organ.sum())
            print('dice = ', dice)
            dice_average[organ_index - 1] += dice
            worksheet.write(organ_index, file_index + 1, dice)
    
    
    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(organ_pred_dir, file.replace('img', 'organ')))
    pred_seg

    speed = time() - start_time

    worksheet.write(5, file_index + 1, speed)

    print('this case use {:.3f} s'.format(speed))
    print('-----------------------')

mean_dice = [dice_organ/10 for dice_organ in dice_average]

worksheet.write(1, len(os.listdir(val_ct_dir))+1, mean_dice[0])
worksheet.write(2, len(os.listdir(val_ct_dir))+1, mean_dice[1])
worksheet.write(3, len(os.listdir(val_ct_dir))+1, mean_dice[2])
worksheet.write(4, len(os.listdir(val_ct_dir))+1, mean_dice[3])
print('mean dice =', mean_dice)

plt.bar(x=organ_list, height=mean_dice, width=0.35)
plt.xlabel('organ')
plt.ylabel('mean dice score')

for organ, dice in enumerate(mean_dice):
    plt.text(organ, dice, '%.3f' % dice, ha='center', va='bottom')

plt.savefig('/output/dice_score/mean_dice_score.png')
# 最后安全关闭表格
workbook.close()

print('==========test end==========')