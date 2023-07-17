import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from .blocks import ConditionalAffineCouplingBlock as CACB
from .modules.squeeze import DCTransform, HaarDownsampling
from .modules.extractors import HIN_CA_Block_
from .modules.transformer import BasicUformerLayer
from utility import *
import torch.nn.init as init
from copy import deepcopy


class Upsample(nn.Module):
    def __init__(self, channels, ratio):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels*ratio*ratio, 1, 1, 0)
        self.ratio = ratio
        self.conv.params_init_scale = 1

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.size()
        x = x.view(B, C // self.ratio // self.ratio, self.ratio, self.ratio, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // self.ratio // self.ratio, H * self.ratio, W * self.ratio)
        return x

class FlowNet(nn.Module):
    def __init__(self, num_blocks, half_layer, heat=0, num_features=31, resolution=64,
                 sample_num=1):
        super(FlowNet, self).__init__()
        self.half_layer = half_layer
        self.condition_blocks = nn.ModuleList()
        num_features_condition = num_features

        self.num_features = num_features
        self.num_features_condition = num_features_condition
        self.heat = heat

        num_features = 90

        self.first_conv = nn.Conv2d(num_features_condition, num_features, 3, 1, 1)
        self.first_conv.params_init_scale = 1

        # self.down = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)
        # self.down.params_init_scale = 1
        self.tr_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        ratio = 1
        for i in range(num_blocks):
            if i in half_layer:
                self.tr_blocks.append(
                    nn.Sequential(
                        # deepcopy(self.down),
                        nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
                        BasicUformerLayer(num_features, (resolution, resolution), 2, 4, 16, 2)
                    )
                )
                self.tr_blocks[-1][0].params_init_scale = 1
                ratio *= 2

            else:
                self.tr_blocks.append(
                    BasicUformerLayer(num_features, (resolution, resolution), 2, 4, 16, 2)
                )
            self.up_blocks.append(
                nn.Sequential(
                    # Upsample(channels=num_features, ratio=ratio),
                    nn.UpsamplingBilinear2d(scale_factor=(ratio, ratio)),
                    HIN_CA_Block_(num_features, num_features_condition*2)
                )
            )
            self.condition_blocks.append(CACB(num_features, num_features_condition, downsampling=False, resolution=resolution, ratio=ratio))

        # self.last_up = Upsample(num_features, ratio)
        self.last_up = nn.UpsamplingBilinear2d(scale_factor=(ratio, ratio))
        self.last_conv = nn.Conv2d(num_features, num_features_condition, 3, 1, 1)
        self.last_conv.params_init_scale = 1

        self.sample_num = sample_num


    def forward(self, inputs):# noisy image, clean image
        if isinstance(inputs, tuple):
            inputs, gt = inputs

            x = inputs
            x = self.first_conv(x)
            cond_list = []
            for tr_block, up_block in zip(self.tr_blocks, self.up_blocks):
                x = tr_block(x)
                cond_list.append([up_block(x)])

            x = self.last_up(x)
            LR = self.last_conv(x)

            zh, logdet = self.encode(gt-LR, cond_list)

            temp = 0
            for i in range(self.sample_num):
                z = self.get_z(inputs, heat=self.heat)
                HF_hat, _ = self.decode(z, cond_list)
                gt_hat = HF_hat + LR
                temp += gt_hat
            gt_hat = temp / self.sample_num

            return gt_hat, (zh, z, logdet), gt_hat
        else:
            x = inputs
            x = self.first_conv(x)
            cond_list = []
            for tr_block, up_block in zip(self.tr_blocks, self.up_blocks):
                x = tr_block(x)
                cond_list.append([up_block(x)])

            x = self.last_up(x)
            LR = self.last_conv(x)

            temp = 0
            for i in range(self.sample_num):
                z = self.get_z(inputs, heat=self.heat)
                HF_hat, logdet = self.decode(z, cond_list)
                gt_hat = HF_hat + LR
                temp += gt_hat
            gt_hat = temp / self.sample_num

            return gt_hat, (z, z, logdet), gt_hat


    def encode(self, z, cond_list=None):
        logdet = 0
        # for i, (block, cond) in enumerate(zip(self.condition_blocks, cond_list)):
        for i, (block, cond) in enumerate(zip(reversed(self.condition_blocks), reversed(cond_list))):
            z, logdet_block = block(z, cond, reverse=False)
            logdet += logdet_block
        return z, logdet

    def decode(self, z, cond_list):
        logdet = 0
        # for i, (block, cond) in enumerate(zip(reversed(self.condition_blocks), reversed(cond_list))):
        for i, (block, cond) in enumerate(zip(self.condition_blocks, cond_list)):
            z, logdet_block = block(z, cond, reverse=True)
            logdet += logdet_block
        return z, logdet



    def get_z(self, LR, mean=0, heat=0):
        B, C, H, W = LR.size()

        z = torch.normal(mean=mean, std=heat,
            size=(B, C, H, W)).to(LR.device)
        # z = torch.from_numpy(np.random.laplace(mean, heat,
        #                 size=(B, C, H, W))).to(LR.device)
        return z

    def init_params(self, init_type, scale):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                if m._get_name() in ['BasicConv3dZeros', 'BasicConv2dZeros']:
                    init.constant_(m.weight, 0)
                elif init_type == 'kn':
                    init.kaiming_normal_(m.weight, mode='fan_in')
                elif init_type == 'ku':
                    init.kaiming_uniform_(m.weight, mode='fan_in')
                elif init_type == 'xn':
                    init.xavier_normal_(m.weight)
                elif init_type == 'xu':
                    init.xavier_uniform_(m.weight)
                if hasattr(m, 'params_init_scale'):
                    m.weight.data *= m.params_init_scale
                else: m.weight.data *= scale
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
