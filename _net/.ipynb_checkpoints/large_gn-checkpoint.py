import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

dropout_rate = 0.5
num_organ = 4
group = 32

# 定义单个3D FCN
class ResUNet_1(nn.Module):
    def __init__(self, training, inchannel, stage):
        """
        :param training: 标志网络是属于训练阶段还是测试阶段
        :param inchannel 网络最开始的输入通道数量
        :param stage 标志网络属于第一阶段，还是第二阶段
        """
        super().__init__()

        self.training = training
        self.stage = stage

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2_2 = nn.Sequential(
            nn.Conv3d(32, 32, (1, 1, 5), 1, padding=(0, 0, 2)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 5, 1), 1, padding=(0, 2, 0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (5, 1, 1), 1, padding=(2, 0, 0)),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )

        self.encoder_stage3_2 = nn.Sequential(
            nn.Conv3d(64, 64, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )

        self.encoder_stage4_2 = nn.Sequential(
            nn.Conv3d(128, 128, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )

        self.encoder_stage5_2 = nn.Sequential(
            nn.Conv3d(256, 256, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(256),
            
            nn.GroupNorm(group, 256),
        )

        self.decoder_stage1_2 = nn.Sequential(
            nn.Conv3d(128 + 128, 128, 1, 1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )

        self.decoder_stage2_2 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 1, 1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )

        self.decoder_stage3_2 = nn.Sequential(
            nn.Conv3d(32 + 32, 32, 1, 1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 1, 5), 1, padding=(0, 0, 2)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 5, 1), 1, padding=(0, 2, 0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (5, 1, 1), 1, padding=(2, 0, 0)),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(16 + 16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        # resolution reduce by half, and double the channels
        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        # keep the resolution and double the channels
        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 2, 2),
            nn.PReLU(256)
        )

        # double the resolution and reduce the channels
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, 2),
            nn.PReLU(16)
        )

        self.map = nn.Sequential(
            nn.Conv3d(16, num_organ + 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2_2(short_range1) + short_range1
        long_range2   = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3_2(short_range2) + short_range2
        long_range3   = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4_2(short_range3) + short_range3
        long_range4   = F.dropout(long_range4, dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs_0 = self.encoder_stage5_2(short_range4) + short_range4
        outputs = F.dropout(outputs_0, dropout_rate, self.training)

        # print('-------decoder-------')

        short_range5 = self.up_conv1(outputs)
        outputs_1 = self.decoder_stage1_2(torch.cat([short_range5, long_range4], dim=1)) + short_range5
        outputs = F.dropout(outputs_1, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs)
        outputs_2 = self.decoder_stage2_2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs_2, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)
        outputs_3 = self.decoder_stage3_2(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs_3, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)
        # print('self.up_conv4 = ', short_range8.shape)

        outputs_4 = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        outputs = self.map(outputs_4)
        # print('self.map = ', outputs.shape)

        return outputs, (long_range1, long_range2, long_range3, long_range4), (
        outputs_4, outputs_3, outputs_2, outputs_1, outputs_0)

class ResUNet_2(nn.Module):
    """
    共9332094个可训练的参数, 九百三十万左右
    """
    def __init__(self, training, inchannel, stage):
        """
        :param training: 标志网络是属于训练阶段还是测试阶段
        :param inchannel 网络最开始的输入通道数量
        :param stage 标志网络属于第一阶段，还是第二阶段
        """
        super().__init__()

        self.training = training
        self.stage = stage

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel + 16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2_2 = nn.Sequential(
            nn.Conv3d(32 + 32, 32, 1, 1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 1, 5), 1, padding=(0, 0, 2)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 5, 1), 1, padding=(0, 2, 0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (5, 1, 1), 1, padding=(2, 0, 0)),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )

        self.encoder_stage3_2 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 1, 1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )

        self.encoder_stage4_2 = nn.Sequential(
            nn.Conv3d(128 + 128, 128, 1, 1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )

        self.encoder_stage5_2 = nn.Sequential(
            nn.Conv3d(256 + 256, 256, 1, 1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(256),
            
            nn.GroupNorm(group, 256),
        )

        self.decoder_stage1_2 = nn.Sequential(
            nn.Conv3d(128 + 128 + 128, 128, 1, 1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )

        self.decoder_stage2_2 = nn.Sequential(
            nn.Conv3d(64 + 64 + 64, 64, 1, 1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 1, 7), 1, padding=(0, 0, 3)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (1, 7, 1), 1, padding=(0, 3, 0)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (7, 1, 1), 1, padding=(3, 0, 0)),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )

        self.decoder_stage3_2 = nn.Sequential(
            nn.Conv3d(32 + 32 + 32, 32, 1, 1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 1, 5), 1, padding=(0, 0, 2)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (1, 5, 1), 1, padding=(0, 2, 0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (5, 1, 1), 1, padding=(2, 0, 0)),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(16 + 16 + 16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 2, 2),
            nn.PReLU(256)
        )

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, 2),
            nn.PReLU(16)
        )

        self.map = nn.Sequential(
            nn.Conv3d(16, num_organ + 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs, long_ranges=None, outputs_ranges=None):
        long_range1  = self.encoder_stage1(torch.cat([inputs, long_ranges[0]], dim=1))
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2_2(torch.cat([short_range1, long_ranges[1]], dim=1)) + short_range1
        long_range2   = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3_2(torch.cat([short_range2, long_ranges[2]], dim=1)) + short_range2
        long_range3   = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4_2(torch.cat([short_range3, long_ranges[3]], dim=1)) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs_0 = self.encoder_stage5_2(torch.cat([short_range4, outputs_ranges[4]], dim=1)) + short_range4
        outputs = F.dropout(outputs_0, dropout_rate, self.training)

#        print('-------decoder-------')
        short_range5 = self.up_conv1(outputs)
        outputs_1  = self.decoder_stage1_2(torch.cat([short_range5, long_range4, outputs_ranges[3]], dim=1)) + short_range5
        outputs      = F.dropout(outputs_1, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs)
        outputs_2  = self.decoder_stage2_2(torch.cat([short_range6, long_range3, outputs_ranges[2]], dim=1)) + short_range6
        outputs      = F.dropout(outputs_2, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)
        outputs_3 = self.decoder_stage3_2(torch.cat([short_range7, long_range2, outputs_ranges[1]], dim=1)) + short_range7
        outputs     = F.dropout(outputs_3, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)
        # print('self.up_conv4 = ', short_range8.shape)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1, outputs_ranges[0]], dim=1)) + short_range8

        outputs = self.map(outputs)
        # print('self.map = ', outputs.shape)

        # 返回概率图
        return outputs


# 定义最终的级连3D FCN
class Net(nn.Module):
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.stage1 = ResUNet_1(training=training, inchannel=1, stage='stage1')
        # num_organ + 2: '1' for background, '1' for initial input
        self.stage2 = ResUNet_2(training=training, inchannel=num_organ + 2, stage='stage2')

    def forward(self, inputs):

        # 得到第一阶段的结果
        output_stage1, long_ranges, outputs_ranges = self.stage1(inputs)
        # after self.stage1(), outputs_stage1 = 1 * 1 * 48 * 128 * 128

        # 将第一阶段的结果与原始输入数据进行拼接作为第二阶段的输入
        inputs_stage2 = torch.cat((output_stage1, inputs), dim=1)
        # inputs_stage2 = 1 * 15 * 48 * 256 * 256

        # 得到第二阶段的结果
        output_stage2 = self.stage2(inputs_stage2, long_ranges, outputs_ranges)
        # output_stage2 = 1 * 14 * 48 * 256 * 256

        if self.training is True:
            return output_stage1, output_stage2
        else:
            return output_stage2


# 网络参数初始化函数
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = Net(training=True)
net.apply(init)

