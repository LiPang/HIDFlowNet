import torch


def get_pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


def split_feature(tensor, coef=None, method='split_feature'):
    if coef is not None:
        C = coef
    else:
        C = tensor.size(1) // 2
    if method == "split_feature":
        return tensor[:, :C, ...], tensor[:, C:, ...]
    elif method == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
    elif method == "horizontal_cross":
        return tensor[:, :, 0::2, ...], tensor[:, :, 1::2, ...]
    elif method == "vertical_cross":
        return tensor[..., 0::2], tensor[..., 1::2]
    else:
        raise ValueError('invalid split_feature method')


def cat_feature(tensor_a, tensor_b, method='split_feature'):
    if method == "split_feature":
        return torch.cat((tensor_a, tensor_b), dim=1)
    elif method == "cross":
        tensor = torch.zeros_like(torch.cat((tensor_a, tensor_b), dim=1))
        tensor[:, 0::2, ...] = tensor_a
        tensor[:, 1::2, ...] = tensor_b
        return tensor
    elif method == "horizontal_cross":
        tensor = torch.zeros_like(torch.cat((tensor_a, tensor_b), dim=2))
        tensor[:, :, 0::2, ...] = tensor_a
        tensor[:, :, 1::2, ...] = tensor_b
        return tensor
    elif method == "vertical_cross":
        tensor = torch.zeros_like(torch.cat((tensor_a, tensor_b), dim=3))
        tensor[..., 0::2] = tensor_a
        tensor[..., 1::2] = tensor_b
        return tensor
    else:
        raise ValueError('invalid split_feature method')