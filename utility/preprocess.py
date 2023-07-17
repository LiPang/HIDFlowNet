from scipy.io import loadmat, savemat
import skimage
import torch
from utility.utils import minmax_normalize
import h5py


def splitWDC():
    imgpath = '../data/WDC/data/dc.tif'
    imggt = skimage.io.imread(imgpath)
    # 转为mat
    imggt = minmax_normalize(imggt)
    imggt = torch.tensor(imggt, dtype=torch.float)

    test = imggt[:, -200:, -200:].clone()
    train_0 = imggt[:, :-200, :].clone()
    train_1 = imggt[:, :, :-200].clone()


    savemat("../data/WDC/data/train_0.mat", {'data': train_0})
    savemat("../data/WDC/data/train_1.mat", {'data': train_1})
    savemat("../data/WDC/data/test.mat", {'data': test})


def readGF2():
    f = h5py.File("/home/cxy/LPang/GlowModels/data/GF2/train/train_gf2.h5", "r")
    data = f['gt'][:]
    return data

if __name__ == '__main__':
    # splitWDC()
    data = readGF2()