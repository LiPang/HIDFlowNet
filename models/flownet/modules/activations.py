import torch

def slog(z: torch.Tensor, alpha=0.1, reverse=False):
    if not reverse:
        return (torch.sign(z) / alpha) * (torch.log(alpha * torch.abs(z) + 1))
    else:
        return (torch.sign(z) / alpha) * (torch.exp(alpha * torch.abs(z)) - 1)