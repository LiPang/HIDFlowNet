import torch.nn as nn
import torch
import torch.nn.functional as F
from .modules.extractors import HIN_CA_Block_
from .modules.affines import SelfLinearAffine, ConditionalLinearAffine
from .modules.permutations import InvertibleConv1x1,InvertibleSpatialConv


class ConditionalAffineCouplingBlock(nn.Module):
    def __init__(self, num_features, num_features_condition=31, downsampling=False, resolution=64, ratio=1):
        super(ConditionalAffineCouplingBlock, self).__init__()
        self.downsampling = downsampling

        self.num_features = num_features
        self.num_features_condition = num_features_condition
        self.permutation = InvertibleConv1x1(num_features_condition, num_features_condition)
        self.cond_feature_extract = ConditionalLinearAffine(num_features, num_features_condition)


    def forward(self, z, cond_list=None, reverse=False):
        logdet_block = torch.zeros_like(z[:, 0, 0, 0])

        if not reverse:

            for cond in cond_list:
                z, logdet = self.cond_feature_extract(z, cond, reverse=False)
                logdet_block += logdet
                z, logdet = self.permutation(z, reverse=False)
                logdet_block += logdet

            outputs = z
            return outputs, logdet_block
        else:

            for cond in reversed(cond_list):
                z, logdet = self.permutation(z, reverse=True)
                logdet_block -= logdet
                z, logdet = self.cond_feature_extract(z, cond, reverse=True)
                logdet_block -= logdet

            return z, logdet_block

