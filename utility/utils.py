import random
import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
import time
import torch.nn.functional as F

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def minmax_normalize(array):
    array = array.astype(np.float)
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def torch2numpy(hsi, use_2dconv):
    if use_2dconv:
        R_hsi = hsi.data[0].cpu().numpy().transpose((1, 2, 0))
    else:
        R_hsi = hsi.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))
    return R_hsi

""" Visualize """
def Visualize3D(data, frame = 0):
    data = np.squeeze(data)
    data[frame, ...] = minmax_normalize(data[frame, ...])
    plt.imshow(data[frame, :, :], cmap='gray')  # shows 256x256 image, i.e. 0th frame
    plt.show()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

term_width = 30
# TOTAL_BAR_LENGTH = 25.
TOTAL_BAR_LENGTH = 1.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')


    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    L.append(' | Remain: %.3f h' % (step_time*(total-current-1) / 3600))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    sys.stdout.write(' %d/%d ' % (current+1, total))
    sys.stdout.flush()


""" learning rate """
def adjust_learning_rate(optimizer, lr):
    print('Adjust Learning Rate => %.4e' %lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def display_learning_rate(optimizer):
    lrs = []
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        print('learning rate of group %d: %.4e' % (i, lr))
        lrs.append(lr)
    return lrs

def harr_downsampling(img: torch.Tensor):
    channel_in = img.shape[1]
    haar_weights = torch.ones((4, 1, 2, 2), requires_grad=False)

    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1

    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1

    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1

    haar_weights = torch.cat([haar_weights] * channel_in, 0).to(img.device)
    out = F.conv2d(img, haar_weights, bias=None, stride=2, groups=channel_in) / 4.0
    out = out.reshape([img.shape[0], channel_in, 4, img.shape[2] // 2, img.shape[3] // 2])
    out = torch.transpose(out, 1, 2)

    out = out.reshape([img.shape[0], channel_in * 4, img.shape[2] // 2, img.shape[3] // 2])
    return out

def harr_upsampling(img: torch.Tensor):
    channel_in = img.shape[1] // 4
    haar_weights = torch.ones((4, 1, 2, 2), requires_grad=False)

    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1

    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1

    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1

    haar_weights = torch.cat([haar_weights] * channel_in, 0).to(img.device)

    out = img.reshape([img.shape[0], 4, channel_in, img.shape[2], img.shape[3]])
    out = torch.transpose(out, 1, 2)
    out = out.reshape([img.shape[0], channel_in * 4, img.shape[2], img.shape[3]])
    return F.conv_transpose2d(out, haar_weights, bias=None, stride=2, groups=channel_in)

# def dct_lr(x, factor=2):
#     coef = dct.dct_3d(x)
#     _, C, H, W = x.shape
#     coef[:, (C + 1) // factor:, :, :] = 0
#     x = dct.idct_3d(coef)
#     return x
#
# def dct_lr3d(x, thr=0.999):
#     coef = dct.dct_3d(x)
#     coef[coef.abs() < torch.quantile(coef.abs(), thr)] = 0
#     x = dct.idct_3d(coef)
#     return x