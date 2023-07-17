import torch
import torch.nn as nn
from .extractors import QRUBlock, Conv3dBlock, Conv2dBlock, HIN_CA_Block, HIN_CA_Block_
from .transformer import LeWinTransformerBlock, BasicUformerLayer

class SelfLinearAffine(nn.Module):
    def __init__(self, in_channel, hidden_channel, clamp=1):
        super(SelfLinearAffine, self).__init__()
        # self.extractor = QRUBlock(in_channels=1, hidden_channels=16)
        # self.extractor = Conv3dBlock(in_channels=1, hidden_channels=16)
        # self.extractor = Conv2dBlock(in_channels=in_channel, hidden_channels=hidden_channel, out_channels=hidden_channel)
        self.extractor = HIN_CA_Block_(in_channel, hidden_channel)
        self.clamp = clamp

    def forward(self, z1, z2, reverse=False):
        if not reverse:
            return self.encode(z1, z2)
        else:
            return self.decode(z1, z2)

    def get_logdet(self, scale):
        return torch.sum(torch.log(scale), dim=[1, 2, 3])

    def tanh(self, scale, eps=1e-5):
        return torch.tanh(scale) + 1 + eps

    def sigmoid(self, scale, clamp=0.5):
        return 1 / (1+torch.exp(-clamp*scale)) + 0.5

    def get_logexpdet(self, scale):
        return torch.sum(self.clamp * 0.636 * torch.atan(scale / self.clamp), dim=[1, 2, 3])

    def exp(self, scale):
        return torch.exp(self.clamp * 0.636 * torch.atan(scale / self.clamp))

    def encode(self, z1, z2):
        scale, shift = self.extractor(z1)
        scale = self.exp(scale)
        z2 = scale * z2 + shift
        logdet = self.get_logdet(scale)
        return z1, z2, logdet

    def decode(self, z1, z2):
        scale, shift = self.extractor(z1)
        scale = self.exp(scale)
        z2 = (z2 - shift) / scale
        logdet = self.get_logdet(scale)
        return z1, z2, logdet


class ConditionalLinearAffine(nn.Module):
    def __init__(self, in_channel, hidden_channel, clamp=1):
        super(ConditionalLinearAffine, self).__init__()
        # self.extractor = QRUBlock(in_channels=1, hidden_channels=16)
        # self.extractor = Conv3dBlock(in_channels=1, hidden_channels=16)
        # self.extractor = HIN_CA_Block(in_channel, hidden_channel*2)
        # self.extractor1 = HIN_CA_Block_(in_channel, hidden_channel)
        # self.extractor2 = HIN_CA_Block_(in_channel, hidden_channel)
        self.clamp = clamp

    def forward(self, z, cond, reverse=False):
        if not reverse:
            return self.encode(z, cond)
        else:
            return self.decode(z, cond)

    def exp(self, scale):
        return torch.exp(self.clamp * 0.636 * torch.atan(scale / self.clamp))

    def get_logdet(self, scale):
        return torch.sum(torch.log(scale), dim=[1, 2, 3])

    def encode(self, z, cond):
        # scale, shift = self.extractor1(cond), self.extractor2(cond)
        # scale, shift = self.extractor(cond)
        scale, shift = cond.chunk(2, 1)
        scale = self.exp(scale)
        z = scale * z + shift
        # z = scale * z
        logdet = self.get_logdet(scale)
        return z, logdet

    def decode(self, z, cond):

        # scale, shift = self.extractor1(cond), self.extractor2(cond)
        # scale, shift = self.extractor(cond)
        scale, shift = cond.chunk(2, 1)
        scale = self.exp(scale)
        z = (z - shift) / scale
        # z = z / scale
        logdet = self.get_logdet(scale)
        return z, -logdet


class TransformerSelfAffine(nn.Module):
    def __init__(self, dim, input_resolution=64, depth=1, num_heads=4, win_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='leff', se_layer=False):
        super(TransformerSelfAffine, self).__init__()
        self.layer = BasicUformerLayer(dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                 drop_path, norm_layer, use_checkpoint,
                 token_projection, token_mlp, se_layer)

    def forward(self, x):
        x = self.layer(x)
        return x