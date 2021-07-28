#coding=UTF-8
 
import os
#from time import time
import time
import argparse

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

cudnn.benchmark = True

parser = argparse.ArgumentParser("The Parameters of the model.")

# Basic parameters
parser.add_argument('--model_name', type=str, default='BLSC', help='training model name')
parser.add_argument('--data_path', type=str, default='/home/zjm/Data/HaN_OAR_256/', help='the root path of data')
parser.add_argument('--loss_image_path', type=str, default='/home/zjm/Project/segmentation/BLSC/loss_show/')
parser.add_argument('--save_path', type=str, default='/home/zjm/Project/segmentation/BLSC/checkpoints/')
parser.add_argument('--val_subset', type=str, default='subset5', help='the directory name of the validation set in data_path, the rest of directory will be used as training set')
parser.add_argument('--repeat_times', type=str, default='2', help='the number of repeated experiments')

parser.add_argument('--load_all_image', action='store_true', help='load all images in ram')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--pin_memory', type=bool, default=False)

# Training parameters
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--learning_rate', type=float, default=1e-4, help='the original learning rate of training')
parser.add_argument('--slice_size', type=int, default=48)
parser.add_argument('--slice_expand', type=int, default=10, help='providing a random range when selecting slices')
parser.add_argument('--num_organ', type=int, default=22)
parser.add_argument('--HU_threshold', type=int, default=1000)

# Resuming training
parser.add_argument('--resume_training', action='store_true', help='Resume training from the pretrained model')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--MultiStepLR', type=int, nargs='+')

args = parser.parse_args()
print(args)
# exit()

def main():
    if args.model_name == 'BLSC':
        from _net.BLSC import net
    elif args.model_name == 'M_BLSC':
        from _net.M_BLSC import net

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    net = torch.nn.DataParallel(net).cuda()

    # Note: loading the state dictionary should be after allocating gpu
    if args.resume_training:
        net.load_state_dict(torch.load(args.resume_model))

    save_path = os.path.join(args.save_path, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.load_all_image:
        cts = []
        gts = []
 
        print("**********Loading all images and labels**********")
        for root, dirs, files in os.walk(args.data_path):
            if (args.val_subset not in root) and 'image.nii' in files:
                ct = sitk.ReadImage(os.path.join(root, 'image.nii'))
                ct_arr = sitk.GetArrayFromImage(ct)
                ct_arr = ct_arr / args.HU_threshold
                cts.append(ct_arr)

                gt = sitk.ReadImage(os.path.join(root, 'label.nii'))
                gt_arr = sitk.GetArrayFromImage(gt)
                gts.append(gt_arr)

        print("**********Loading end**********")

    train_ds = Dataset(args.slice_size, cts=cts, gts=gts, slice_expand=args.slice_expand)
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=args.pin_memory)

    loss_func = DiceLoss()
    #loss_func = CELoss()

    opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, args.MultiStepLR)

    start = time.time()
    loss_list = []
    min_loss = 10
    for epoch in range(args.epochs):
        mean_loss = []

        for step, (ct_list, seg_list) in enumerate(train_dl):

            for index in range(len(ct_list)):
                ct = ct_list[index]
                seg = seg_list[index]
                ct = ct.cuda()

                if args.model_name == 'UNet' or args.model_name == 'VNet':
                    outputs = net(ct)
                    loss    = loss_func(outputs, seg, args.num_organ, args.slice_size)
                else:
                    outputs_stage1, outputs_stage2 = net(ct)
                    loss = loss_func(outputs_stage1, outputs_stage2, seg, args.num_organ, args.slice_size)

                mean_loss.append(loss.item())

                opt.zero_grad()

                loss.backward()
                opt.step()

            if step % 5 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                    .format(epoch, step, loss.item(), (time.time() - start) / 60))

        lr_decay.step()
        mean_loss = sum(mean_loss) / len(mean_loss)
        print('mean_loss = ', mean_loss)
        loss_list.append(mean_loss)

        if mean_loss < min_loss:
            min_loss = mean_loss
            torch.save(net.state_dict(), save_path + 'best_' + args.model_name + args.repeat_times + '.pth')
        print('min_loss = ', min_loss)

        # check the time and draw the loss image every 10 epochs
        if epoch % 5 == 0:
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

            x = range(0, epoch + 1)
            plt.plot(x, loss_list, 'o-')
            plt.title('Loss with epochs')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.savefig('/home/zjm/Project/segmentation/BLSC/loss_show/' + args.model_name + args.repeat_times + '.png')

        print('************************')
        print()

    print('min_loss = ', min_loss)

if __name__ =='__main__':
    main()