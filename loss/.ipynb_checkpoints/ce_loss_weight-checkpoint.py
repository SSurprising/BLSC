"""

基本的交叉熵损失函数
"""
import torch
import torch.nn as nn
import numpy as np


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_stage1, pred_stage2, target, num_organ=4):
        """

        :param pred_stage1: (B, 14, 48, 256, 256)
        :param target: (B, 48, 256, 256)

        """
        organ_w = np.zeros(num_organ + 1)
        num = np.zeros(num_organ + 1)
        N_total = target.shape[0] * target.shape[1] * target.shape[2] * target.shape[3]

        unique, counts = np.unique(target, return_counts=True)
        unique = [int(a) for a in unique]

        for i in range(len(unique)):
            num[int(unique[i])] = counts[i]
#         print(num)
        
        for i in range(num_organ + 1):
            organ_w[i] = (1 - num[i]/N_total) / num_organ
#         print('organ_w = ', organ_w)
        # 计算交叉熵损失值
        organ_w = torch.Tensor(organ_w).cuda()
        target = torch.Tensor(target).cuda().long()
        self.loss = nn.CrossEntropyLoss(weight=organ_w)
        
        loss_stage1 = self.loss(pred_stage1, target)
        loss_stage2 = self.loss(pred_stage2, target)
        
        # 最终的损失值由两部分组成
        loss = (loss_stage1 + loss_stage2) / 2

        return loss
