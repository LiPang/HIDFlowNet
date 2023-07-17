import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return F.mse_loss(x, target)
        elif self.losstype == 'l1':
            return F.l1_loss(x, target)
        elif self.losstype == 'l1_smooth':
            return F.smooth_l1_loss(x, target)
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return torch.mean(torch.mean(-torch.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


def get_pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


class NegativeLikelihoodLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super(NegativeLikelihoodLoss, self).__init__()
        self.sigma = sigma
        self.gauss_constant = 0.5 * float(np.log(2 * np.pi)) + np.log(sigma) # Gauss
        # self.laplace_constant = np.log(2 * sigma) # Laplace

    def forward(self, z, logdet):
        objective = torch.sum(-z ** 2 / (2 * self.sigma ** 2) - self.gauss_constant, dim=[1, 2, 3]) / float(
            z.size(1) * get_pixels(z))
        # objective = torch.sum(-torch.abs(z) / self.sigma - self.laplace_constant, dim=[1, 2, 3]) / float(
        #     z.size(1) * get_pixels(z))
        objective = torch.mean(objective)
        logdet = logdet / float(z.size(1) * get_pixels(z))
        logdet = torch.mean(logdet)
        nll = -(objective + logdet)
        return nll



class BiReconstructionLoss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=1, gamma=0.001, sigma=0.1):
        super(BiReconstructionLoss, self).__init__()
        self.Reconstruction_back = ReconstructionLoss(losstype='l1')
        self.Distribution_loss = NegativeLikelihoodLoss(sigma)
        self.gamma = gamma
        self.lambda_fit_forw = lambda1
        self.lambda_rec_back = lambda2
        self.sigma = sigma
        self.Log2PI = 0.5 * float(np.log(2 * np.pi)) + np.log(sigma)

    def loss_forward(self, out, y):
        l_forw_fit = self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_backward(self, x, y):
        l_back_rec = self.Reconstruction_back(x, y)
        return l_back_rec

    def loss_geom(self, x, y):
        x, y = x.flatten(1, 3), y.flatten(1, 3)
        l_dis = self.Distribution_loss(x, y).mean()
        return l_dis

    def loss_nll(self, x, y):
        l_dis = self.Distribution_loss(x, y)
        return l_dis

    def forward(self, predict, target):
        gt_hat, outputs, LR_hat = predict
        loss_backward = self.lambda_rec_back * self.loss_backward(gt_hat, target)
        loss_dis = self.gamma * self.loss_nll(outputs[0], outputs[2])
        return loss_backward + loss_dis