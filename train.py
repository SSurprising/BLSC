#coding=UTF-8
 
import os
#from time import time
import time
 
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
import SimpleITK as sitk

from loss.ava_Dice_loss_with_bg import DiceLoss
from dataset.dataset2 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
model_name = 'BLSC'
val_subset = 'subset5'
times = '1'
restore_flag = False

if model_name == 'BLSC':
    from _net.BLSC import net
elif model_name == 'M_BLSC':
    from _net.M_BLSC import net

# 定义超参数
on_server = False
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
Epoch = 2000

leaing_rate = 1e-4
 

cudnn.benchmark = True
batch_size = 1 if on_server is False else 3
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True
 
net = torch.nn.DataParallel(net).cuda()


slice_size = 48
slice_expand = 10
num_organ = 22
file_dir = '/home/zjm/Data/HaN_OAR_256/'

loss_image_path = '/output/loss_show/'
save_path = r'/home/zjm/Project/segmentation/BLSC/checkpoints/' + model_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

load_all_CT = True
if load_all_CT:
    cts   = []
    gts = []
 
    print("**********Loading all images and labels**********")
    for root, dirs, files in os.walk(file_dir):
        if (val_subset not in root) and 'image.nii' in files:

            ct = sitk.ReadImage(os.path.join(root, 'image.nii'))
            ct_arr = sitk.GetArrayFromImage(ct)
            cts.append(ct_arr)

            gt = sitk.ReadImage(os.path.join(root, 'label.nii'))
            gt_arr = sitk.GetArrayFromImage(gt)
            gts.append(gt_arr)

    print("**********Loading end**********")

# 定义数据加载
train_ds = Dataset(slice_size, cts=cts, gts=gts, slice_expand=slice_expand)
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = DiceLoss()
#loss_func = CELoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [700, 1000, 1500])

# 训练网络
start = time.time()
loss_list = []
min_loss = 10
for epoch in range(Epoch):

    lr_decay.step()

    mean_loss = []

    for step, (ct_list, seg_list) in enumerate(train_dl):

        for index in range(len(ct_list)):
            ct = ct_list[index]
            seg = seg_list[index]
            ct = ct.cuda()
            ct /= 1000

            if model_name == 'UNet' or model_name == 'VNet':
                outputs = net(ct)
                loss    = loss_func(outputs, seg, num_organ, slice_size)
            else:
                outputs_stage1, outputs_stage2 = net(ct)
                loss = loss_func(outputs_stage1, outputs_stage2, seg, num_organ, slice_size)

            mean_loss.append(loss.item())

            opt.zero_grad()

            loss.backward()
            opt.step()

        if step % 5 == 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time.time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)
    print('mean_loss = ', mean_loss)

    loss_list.append(mean_loss)
    x = range(0, epoch + 1)
    plt.plot(x, loss_list, 'o-')
    plt.title('Loss with epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('/home/zjm/Project/segmentation/BLSC/loss_show/' + model_name + times + '.png')

    if mean_loss < min_loss:
        min_loss = mean_loss
        torch.save(net.state_dict(), save_path + 'best_' + model_name + times + '.pth')
    print('min_loss = ', min_loss)

    # check the time every 10 epochs
    if epoch % 10 == 0:
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    print('************************')
    print()



print('min_loss = ', min_loss)