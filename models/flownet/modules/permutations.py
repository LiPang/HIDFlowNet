import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
try: from .utils import get_pixels
except: from models.flownet2.modules.utils import get_pixels
# class InvertibleSpatialConv(nn.Module):
#     def __init__(self, K):
#         super(InvertibleSpatialConv, self).__init__()
#         self.K = K
#         w_shape = [K ** 2, K ** 2]
#         w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
#         w_init = nn.Parameter(torch.Tensor(w_init))
#         self.weight = w_init
#
#     def forward(self, input, reverse):
#         B, C, H, W = input.shape
#         K = self.K
#         dlogdet = torch.slogdet(self.weight)[1] * (H // K * W // K * C)
#
#         patches = input.reshape(B, C, H // K, K, W // K, K)  # B C H//K K1 W//K K2
#         patches = patches.permute(0, 1, 2, 4, 3, 5)  # B C H//K W//K K1 K2
#         patches = patches.reshape(B, C * H // K * W // K, K ** 2)  # B*C H//K * W//K K1*K2
#         if not reverse:
#             weight = self.weight
#             patches = F.linear(patches, weight)
#
#             patches = patches.reshape(B, C, H // K, W // K, K, K)  # B C H//K W//K K1 K2
#             patches = patches.permute(0, 1, 2, 4, 3, 5)  # B C H//K K1 W//K K2
#             patches = patches.reshape(B, C, H, W)
#             return patches, dlogdet
#         else:
#             weight = torch.inverse(self.weight.double()).float()
#             patches = F.linear(patches, weight)
#
#             patches = patches.reshape(B, C, H // K, W // K, K, K)  # B C H//K W//K K1 K2
#             patches = patches.permute(0, 1, 2, 4, 3, 5)  # B C H//K K1 W//K K2
#             patches = patches.reshape(B, C, H, W)
#             return patches, -dlogdet

def fft_shift_matrix(n, shift_amount):
    shift = torch.arange(0, n).repeat((n, 1))
    shift = shift + shift.T
    return torch.exp(1j * 2 * np.pi * shift_amount * shift / n)

# class InvertibleSpatialConv(nn.Module):
#     def __init__(self, channel, kernel_size):
#         super(InvertibleSpatialConv, self).__init__()
#         self.k = kernel_size
#         w_shape = [kernel_size, kernel_size]
#
#         function_totensor = lambda x: torch.Tensor(x[None, None, ...])
#
#         w_init = [np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32) * 0.1
#                   for _ in range(channel)]
#         self.weight = torch.nn.Parameter(torch.cat(list(map(function_totensor, w_init)), dim=1))
#
#     def get_logdet(self):
#         return 0
#
#     def conv(self, x, weight, n, k):
#         wpad = F.pad(weight, (0, n - k, 0, n - k))
#         shift_amount = (k - 1) // 2
#         shift_matrix = fft_shift_matrix(n, -shift_amount).to(x.device)
#         wfft = shift_matrix * torch.fft.fft2(wpad).conj()
#
#         xfft = torch.fft.fft2(x)
#         yfft = wfft * xfft
#         y = torch.real(torch.fft.ifft2(yfft))
#         return y
#
#     def inv_conv(self, x, weight, n, k):
#         wpad = F.pad(weight, (0, n - k, 0, n - k))
#         shift_amount = (k - 1) // 2
#         shift_matrix = fft_shift_matrix(n, -shift_amount).to(x.device)
#         wfft = shift_matrix * torch.fft.fft2(wpad).conj()
#
#         yfft = torch.fft.fft2(x)
#         xfft = yfft / wfft
#         x = torch.real(torch.fft.ifft2(xfft))
#         return x
#
#     def forward(self, x, reverse):
#         k, n, c = self.k, x.shape[-1], x.shape[1]
#         x_list, weight_list = x.split(1,1), self.weight.split(1,1)
#
#         if not reverse:
#             y = list(map(self.conv, x_list, weight_list, [n]*c, [k]*c))
#             y = torch.cat(y, dim=1)
#             logdet = self.get_logdet()
#             return y, logdet
#         else:
#             y = list(map(self.inv_conv, x_list, weight_list, [n]*c, [k]*c))
#             y = torch.cat(y, dim=1)
#             logdet = self.get_logdet()
#             return y, -logdet

class InvertibleSpatialConv(nn.Module):
    def __init__(self, channel, kernel_size=2):
        super(InvertibleSpatialConv, self).__init__()        
        w_shape = [kernel_size**2, kernel_size**2]
        w_init = np.eye(kernel_size**2).astype(np.float32)
        w_init = w_init.reshape(kernel_size**2, 1, kernel_size, kernel_size)

        self.weights = torch.cat([torch.from_numpy(w_init)] * channel, 0)
        self.weights = nn.Parameter(self.weights)
        self.channel_in = channel
        self.w_shape = w_shape

        self.inited = True
        
    def forward(self, x, reverse=False):
        if not reverse:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            weight_ = self.weights[:4, ...].reshape(self.w_shape)
            self.last_jac = self.elements / 4 * torch.slogdet(weight_)[1]
            out = F.conv2d(x, self.weights, bias=None, stride=2, groups=self.channel_in)

            return out, self.last_jac
        else:
            out = x
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)
            out = F.conv_transpose2d(out, self.weights, bias=None, stride=2, groups = self.channel_in)
            return out, self.last_jac


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32) * 0.1
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.inited = True

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = get_pixels(input)
        weight = self.weight + torch.eye(w_shape[0], device=self.weight.device)
        dlogdet = torch.slogdet(weight)[1] * pixels  # log(abs(det(self.weight)))
        if not reverse:
            weight = weight.view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            weight = torch.inverse(weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
            return weight, -dlogdet

    def forward(self, input, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        z = F.conv2d(input, weight)
        return z, dlogdet

if __name__ == '__main__':
    # permutation = InvertibleConv1x1(31)
    permutation = InvertibleSpatialConv(31)
    x = torch.randn((1, 31, 64, 64))
    y = permutation(x)
    print('ok')