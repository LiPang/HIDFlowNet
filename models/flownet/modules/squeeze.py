import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import torch_dct as dct

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super(SqueezeLayer, self).__init__()
        self.factor = factor

    def forward(self, inputs, logdet_block=0, reverse=False):
        factor = self.factor
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return inputs, logdet_block
        if not reverse:
            inputs = self.encode(inputs, factor)
            return inputs, logdet_block
        else:
            inputs = self.decode(inputs, factor)
            return inputs, logdet_block

    def encode(self, tensor, factor):
        B, C, H, W = tensor.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = tensor.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x

    def decode(self, tensor, factor):
        B, C, H, W = tensor.size()
        assert C % factor == 0, "{}".format(C)
        x = tensor.view(B, C // factor // factor, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // factor // factor, H * factor, W * factor)
        return x



class HaarDownsampling(nn.Module):
    def __init__(self, channel_in, img_only=False):
        super(HaarDownsampling, self).__init__()
        self.img_only = img_only
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, reverse=False):
        if not reverse:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            if self.img_only:
                return out
            return out, self.last_jac
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            if self.img_only:
                return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in), self.last_jac


# class DCTransform(nn.Module):
#     def __init__(self, factor=2, img_only=False):
#         super(DCTransform, self).__init__()
#         self.img_only = img_only
#         self.factor = factor
#
#     def forward(self, x, reverse=False):
#         if not reverse:
#             coef = dct.dct_2d(x)
#             _, _, H, W = x.shape
#             mask = torch.ones((H, W), device=x.device)
#             mask[:H//self.factor, :W//self.factor] = 0
#             x_lr = dct.idct_2d(coef * (1 - mask))
#             if self.img_only:
#                 return x_lr
#             return x_lr, 0
#         else:
#             if self.img_only:
#                 return x
#             return x, 0
#
# class DCTransform(nn.Module):
#     def __init__(self, factor=2, img_only=False):
#         super(DCTransform, self).__init__()
#         self.img_only = img_only
#         self.factor = factor
#
#     def forward(self, x, reverse=False):
#         if not reverse:
#             coef = dct.dct_2d(x)
#             _, _, H, W = x.shape
#             mask = torch.ones((H, W), device=x.device)
#             mask[:H//self.factor, :W//self.factor] = 0
#             x_lr = dct.idct_2d(coef * (1 - mask))
#             x_hr = dct.idct_2d(coef * mask)
#             x = torch.cat((x_lr, x_hr), dim=1)
#             if self.img_only:
#                 return x
#             return x, 0
#         else:
#             x_lr, x_hr = x.chunk(chunks=2, dim=1)
#             coef_lr, coef_hr = dct.dct_2d(x_lr), dct.dct_2d(x_hr)
#             coef = coef_lr + coef_hr
#             x = dct.idct_2d(coef)
#             if self.img_only:
#                 return x
#             return x, 0

# class DCTransform(nn.Module):
#     def __init__(self, factor=2, img_only=False):
#         super(DCTransform, self).__init__()
#         self.img_only = img_only
#         self.factor = factor
#
#     def forward(self, x, reverse=False):
#         if not reverse:
#             coef_lr, coef_hr = dct.dct_3d(x).chunk(chunks=2, dim=1)
#             x_lr = dct.idct_3d(coef_lr)
#             x_hr = dct.idct_3d(coef_hr)
#             x = torch.cat((x_lr, x_hr), dim=1)
#             if self.img_only:
#                 return x
#             return x, 0
#         else:
#             x_lr, x_hr = x.chunk(chunks=2, dim=1)
#             coef_lr, coef_hr = dct.dct_3d(x_lr), dct.dct_3d(x_hr)
#             coef = torch.cat((coef_lr, coef_hr), dim=1)
#             x = dct.idct_3d(coef)
#             if self.img_only:
#                 return x
#             return x, 0

class DCTransform(nn.Module):
    def __init__(self, factor=2, img_only=False):
        super(DCTransform, self).__init__()
        self.img_only = img_only
        self.factor = factor
        self.thr = 0.999

    def forward(self, x, reverse=False):
        if not reverse:
            coef = dct.dct_3d(x)
            _, C, H, W = x.shape
            # mask = coef.abs() > torch.quantile(coef.abs(), self.thr)
            mask = torch.ones_like(x, device=x.device)
            mask[:, C//2:, ...] = 0
            x_lr = dct.idct_3d(coef * mask)
            x_hr = dct.idct_3d(coef * (1-mask))
            x = torch.cat((x_lr, x_hr), dim=1)
            if self.img_only:
                return x
            return x, 0
        else:
            x_lr, x_hr = x.chunk(chunks=2, dim=1)
            coef_lr, coef_hr = dct.dct_3d(x_lr), dct.dct_3d(x_hr)
            coef = coef_lr + coef_hr
            x = dct.idct_3d(coef)
            if self.img_only:
                return x
            return x, 0

if __name__ == '__main__':
    haar = HaarDownsampling(31)
    x = torch.randn((1, 31, 64, 64))
    y = haar(x)
    print('ok')