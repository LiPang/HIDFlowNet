import torch
import torch.nn as nn
from .conv_units import BasicConv3d, BasicConv3dZeros, BasicConv2d, BasicConv2dZeros
from .utils import split_feature

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='lrelu'):
        super(Conv3dBlock, self).__init__()
        if act == 'lrelu':
            act = nn.LeakyReLU
        self.net = nn.Sequential(
            BasicConv3d(in_channels, hidden_channels, k, s, p, bn=bn),
            act(),
            BasicConv3d(hidden_channels, hidden_channels, k, s, p, bn=bn),
            act(),
            BasicConv3dZeros(hidden_channels, 2, k, s, p, bn=bn)
        )


    def forward(self, inputs, cond=None, reverse=False):
        inputs = inputs[:, None, ...]
        outputs = self.net(inputs)
        scale, shift = outputs[:, 0, :, :], outputs[:, 1, :, :]
        scale = torch.sigmoid(scale) + 0.5
        scale, shift = scale.squeeze(1), shift.squeeze(1)
        return scale, shift

class HinResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HinResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class HIN_CA_Block_(nn.Module):
    def __init__(self, channel_in, channel_out) -> None:
        super().__init__()
        self.hin = HinResBlock(channel_in, channel_out)
        self.ca = CALayer(channel_out, reduction=4)

    def forward(self, x):
        hin_out = self.hin(x)
        ca_out = self.ca(hin_out)
        return ca_out

class HIN_CA_Block(nn.Module):
    def __init__(self, channel_in, channel_out) -> None:
        super().__init__()
        self.hin = HinResBlock(channel_in, channel_out)
        self.ca = CALayer(channel_out, reduction=4)

    def forward(self, x):
        hin_out = self.hin(x)
        ca_out = self.ca(hin_out)
        shift, scale = split_feature(ca_out)
        return scale, shift

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_hidden=3, n_hidden_layers=1):
        super(Conv2dBlock, self).__init__()
        layers = nn.ModuleList()
        layers.append(BasicConv2d(in_channels, hidden_channels))
        layers.append(nn.LeakyReLU())

        for _ in range(n_hidden_layers):
            layers.append(BasicConv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.LeakyReLU())
        layers.append(BasicConv2dZeros(hidden_channels, out_channels))
        self.layers = layers

    def forward(self, inputs, cond=None, reverse=False):
        x = inputs
        for l in self.layers:
            x = l(x)
        shift, scale = split_feature(x, "cross")
        scale = torch.sigmoid(scale) + 0.5
        return scale, shift


class QRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh', eps=1e-4):
        super(QRUBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = BasicConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn)
        self.conv_scale = BasicConv3dZeros(hidden_channels//2, 1, k, s, p, bn=bn)
        self.conv_shift = BasicConv3dZeros(hidden_channels//2, 1, k, s, p, bn=bn)
        self.act = act
        self.eps = eps

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, cond=None, reverse=False):
        inputs = inputs[:, None, ...]
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []

        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                    reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)
        # concatenated hidden states
        h_time = torch.cat(h_time, dim=2)

        # compute scale and shift
        log_scale, shift = h_time[:, :self.hidden_channels//2, ...], h_time[:, self.hidden_channels//2:, ...]
        log_scale, shift = self.conv_scale(log_scale), self.conv_shift(shift)
        scale = torch.sigmoid(log_scale + 2.) + self.eps
        scale, shift = scale.squeeze(1), shift.squeeze(1)
        return scale, shift

