# ------------------------------------------------------------------------------
# Creat on 2022.01.06
# Written by Liu Tao (LiuTaobbu@163.com)
# ------------------------------------------------------------------------------
import numpy as np
import torch

# 实现方式1
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 实现方式2，label也进行mixup
def onehot(ind, num_classes):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)

def soft_cross_entropy(logits, label, reduce_mean=True):
    # log(e^x1+e^x2)
    logp = logits - torch.logsumexp(logits, -1, keepdim=True)
    ce = -1 * torch.sum(label * logp, -1)
    if reduce_mean is True:
        return torch.mean(ce)
    else:
        return ce

def shuffle_minibatch(inputs, targets, mixup_alpha, device):
    if mixup_alpha == 0.:
        return inputs, targets
    mixup_alpha = mixup_alpha

    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]

    ma = np.random.beta(mixup_alpha, mixup_alpha, [batch_size, 1])
    ma_img = ma[:, :, None, None]

    inputs1 = inputs1 * torch.from_numpy(ma_img).to(device).float()
    inputs2 = inputs2 * torch.from_numpy(1 - ma_img).to(device).float()

    targets1 = targets1.float() * torch.from_numpy(ma).to(device).float()
    targets2 = targets2.float() * torch.from_numpy(1 - ma).to(device).float()

    inputs_shuffle = (inputs1 + inputs2).to(device)
    targets_shuffle = (targets1 + targets2).to(device)

    return inputs_shuffle, targets_shuffle