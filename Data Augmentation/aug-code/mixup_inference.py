# ------------------------------------------------------------------------------
# Creat on 2022.01.14
# Written by Liu Tao (LiuTaobbu@163.com)
# ------------------------------------------------------------------------------
import torch
import numpy as np
import torchvision
import torch.nn as nn


num_classes = 7
device = torch.device("cuda:0")
num_pool = 1000 # 数据池的大小
num_sample = 5 # 取样次数
m_lamda = np.random.beta(0.2, 0.2)

# label的one-hot形式转换
def onehot(ind, num_classes):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)
# 分类器
def classifier():
    pass

# 构建数据加载
trainset = torchvision.datasets.CIFAR10(root='./data/cifar/',
                                            train=True,
                                            download=True,
                                            transform=None,
                                            target_transform=onehot)

testset = torchvision.datasets.CIFAR10(root='./data/cifar/',
                                        train=False,
                                        download=True,
                                        transform=None,
                                        target_transform=onehot)

dataloader_train = torch.utils.data.DataLoader(trainset,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=8,
                                               drop_last=True)

dataloader_test = torch.utils.data.DataLoader(testset,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=8,
                                              drop_last=True)

# 构建一个数据池，用于mixup的取用
mixup_pool = {}
for i in range(num_classes):
    mixup_pool.update({i: []})

for i, data_batch in enumerate(dataloader_train):
    # for i, data_batch in enumerate(dataloader_test):
    img_batch, label_batch = data_batch
    img_batch = img_batch.to(device)
    _, label_ind = torch.max(label_batch.data, 1)
    mixup_pool[label_ind.numpy()[0]].append(img_batch)
    if i >= (num_pool - 1):
        break
print('Finish constructing mixup_pool')

# PL方式，与预测标签一致
for i, data_batch in enumerate(dataloader_test):
    img_batch, label_batch = data_batch
    img_batch, label_batch = img_batch.to(device), label_batch.to(device)
    # 原样本的推理
    pred_cle = classifier(img_batch)
    # 概率与类别
    cle_con, predicted_cle = torch.max(nn.Softmax(pred_cle.data, dim=-1), 1)
    predicted_cle = predicted_cle.cpu().numpy()[0]
    # 开始从数据池中取数据进行mixup
    for i in range(num_sample):
        # 随机取样-与原样本推理类别一致
        len_cle = np.random.randint(len(mixup_pool[predicted_cle]))
        mixup_img_cle = (1 - m_lamda) * mixup_pool[predicted_cle][
            len_cle] + m_lamda * img_batch
        pred_cle_mixup = classifier(mixup_img_cle)
        pred_cle_mixup_all = pred_cle_mixup_all + nn.Softmax(pred_cle_mixup.data, dim=-1)
    # 取平均
    pred_cle_mixup_all = pred_cle_mixup_all / num_sample

    cle_con_mixup, predicted_cle_mixup = torch.max(pred_cle_mixup_all, 1)

# OL方式，随机取标签
xs_cle_label = np.random.randint(num_classes)
while xs_cle_label == label_batch:
    xs_cle_label = np.random.randint(num_classes)
