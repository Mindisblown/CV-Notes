# ------------------------------------------------------------------------------
# Creat on 2022.10.27
# Written by Liu Tao (LiuTaobbu@163.com)
# ------------------------------------------------------------------------------
import numpy as np
import torch

def mix_style(x, alpha, eps=1e-6, mix="random"):
    # x input tensor
    batch_size = x.size(0)
    # mean value
    mean_value = x.mean(dim=[2, 3], keepdim=True)
    # variance value
    var_value = x.var(dim=[2, 3], keepdim=True)
    # standard deviation
    sigma_value = (var_value + eps).sqrt()

    # detach gradient
    mean_value = mean_value.detach()
    var_value = var_value.detach()
    # (x - mean) / standard deviation
    x_normal = (x - mean_value) / sigma_value

    lmda = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1, 1, 1))
    lmda = lmda.to(x.device)

    if mix == "random":
        perm = torch.randperm(batch_size)
    elif mix == "crossdomain":
        perm = torch.arange(batch_size - 1, -1, -1)
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(batch_size // 2)]
        perm_a = perm_a[torch.randperm(batch_size // 2)]
        perm = torch.cat([perm_b, perm_a], 0)
    else:
        raise NotImplementedError

    mean_value_2, sigma_value_2 = mean_value[perm], sigma_value[perm]
    mean_mix = mean_value * lmda + mean_value_2 * (1 - lmda)
    sigma_mix = sigma_value * lmda + sigma_value_2 * (1- lmda)

    return x_normal * sigma_mix + mean_mix
