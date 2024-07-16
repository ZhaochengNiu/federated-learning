#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
# 导入所需的库：numpy 用于数学运算，torchvision.datasets 和 torchvision.transforms 用于处理图像数据集。


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # 定义了 mnist_iid 函数，它接收两个参数：
    # dataset：MNIST 数据集的实例。
    # num_users：要分配数据的用户数量。
    # 这是一个多行字符串（docstring），
    # 用于说明函数的功能：从 MNIST 数据集中采样 I.I.D. 客户端数据，以及参数和返回值的描述。
    num_items = int(len(dataset)/num_users)
    # 计算每个用户可以分配到的数据项（图像）数量，这里简单地将数据集的总大小除以用户数量。
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # 初始化一个空字典 dict_users 来存储每个用户的数据索引，以及一个包含数据集中所有索引的列表 all_idxs。
    for i in range(num_users):
        # 遍历从 0 到 num_users-1 的用户。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 对于每个用户，使用 np.random.choice 从 all_idxs 中随机选择 num_items 个不重复的索引，
        # 并将这些索引作为集合赋值给 dict_users[i]。
        all_idxs = list(set(all_idxs) - dict_users[i])
        # 从 all_idxs 中移除已经被分配给当前用户的数据索引，以确保数据的独立同分布性。
    return dict_users
    # 返回包含每个用户数据索引的字典。


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 定义了 mnist_noniid 函数，它接收两个参数：
    # dataset：MNIST 数据集的实例。
    # num_users：要分配数据的用户数量。
    # 这是一个多行字符串（docstring），
    # 用于说明函数的功能：从 MNIST 数据集中采样非 I.I.D. 客户端数据，以及参数和返回值的描述。
    num_shards, num_imgs = 200, 300
    # 设定数据被分成的块（shard）数量和每个块中的图像数量。这里假设数据集被分成200个块，每块包含300张图像。
    idx_shard = [i for i in range(num_shards)]
    # 创建一个列表，包含从0到 num_shards-1 的索引，代表每个数据块的索引。
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # 为每个用户初始化一个空的 NumPy 数组，用于存储分配给该用户的数据索引。
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
