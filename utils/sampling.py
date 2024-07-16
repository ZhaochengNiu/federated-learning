#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
# 导入所需的库：numpy 用于数学运算，torchvision.datasets 和 torchvision.transforms 用于处理图像数据集。

# 导入所需的库：NumPy 用于数学运算
# torchvision.datasets 和 torchvision.transforms 用于加载和处理图像数据集


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
        # """
        # Sample I.I.D. client data from MNIST dataset
        # :param dataset: MNIST 数据集的实例。
        # :param num_users: 要分配数据给的用户数量。
        # :return: 一个字典，包含每个用户的数据索引。
        # """
    num_items = int(len(dataset)/num_users)
    # 计算每个用户应该获得的数据项（图像）数量。这里通过将数据集的总大小除以用户数量来实现，结果转换为整数。
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # 初始化两个变量：dict_users 是一个空字典，用于存储每个用户的数据索引；all_idxs 是一个列表，包含数据集中所有图像的索引。
    for i in range(num_users):
        # 开始一个循环，遍历每个用户。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 对于每个用户，使用 np.random.choice 函数从 all_idxs 中随机选择 num_items 个不重复的索引，
        # 并将这些索引作为一个集合赋值给 dict_users[i]。
        all_idxs = list(set(all_idxs) - dict_users[i])
        # 从 all_idxs 中移除已经被当前用户选中的索引，以确保数据的独立同分布性，即每个用户获得的数据是完全随机且不重复的。
    return dict_users
    # 在为所有用户分配了数据索引后，返回包含这些索引的字典 dict_users。


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 定义了 mnist_noniid 函数，它接受两个参数：dataset 是一个 MNIST 数据集的实例，num_users 是参与的用户数。
    # 函数的返回值是一个字典，其中包含每个用户的数据索引数组。
    num_shards, num_imgs = 200, 300
    # 设定数据被分成的块（shard）数量为 200，每个块中的图像数量为 300。
    # 这里假设整个数据集有 200 个块，每块包含 300 张图像。
    idx_shard = [i for i in range(num_shards)]
    # 创建一个列表，包含从 0 到 num_shards-1 的索引，代表每个数据块的索引。
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # 为每个用户初始化一个空的 NumPy 数组，用于存储分配给该用户的数据索引。
    idxs = np.arange(num_shards*num_imgs)
    # 创建一个数组，包含从 0 到 num_shards*num_imgs-1 的所有数据索引。
    labels = dataset.train_labels.numpy()
    # 获取训练数据集的标签，并将其转换为 NumPy 数组。
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # 将数据索引和标签垂直堆叠成一个新的数组。
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # 根据标签对 idxs_labels 进行排序，确保相同标签的数据排在一起。
    idxs = idxs_labels[0,:]
    # 更新 idxs 数组，使其按标签排序。
    # divide and assign
    for i in range(num_users):
        # 开始一个循环，遍历每个用户。
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # 为当前用户随机选择 2 个不同的数据块索引，放入 rand_set 集合。replace=False 确保选择的索引不重复。
        idx_shard = list(set(idx_shard) - rand_set)
        # 从 idx_shard 列表中移除已经被当前用户选中的数据块索引。
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # 对于用户 i 选中的每个数据块索引 rand，将该数据块内的所有数据索引添加到 dict_users[i] 中。
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #   """
    #    从 CIFAR10 数据集采样 I.I.D. 客户端数据
    #    :param dataset: CIFAR10 数据集的实例。
    #    :param num_users: 要分配数据给的用户数量。
    #    :return: 包含每个用户图像索引的字典。
    #   """
    num_items = int(len(dataset)/num_users)
    # 计算每个用户应该获得的数据项（图像）数量，通过将数据集的总大小除以用户数量得到，结果转换为整数。
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # 初始化两个变量：dict_users 是一个空字典，
    # 用于存储每个用户的数据索引集合；all_idxs 是一个列表，包含数据集中所有图像的索引。
    for i in range(num_users):
        # 开始一个循环，遍历每个用户，为每个用户分配数据。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 对于每个用户 i，使用 np.random.choice 函数从 all_idxs 中随机选择 num_items 个不重复的索引，
        # 并创建一个索引集合赋值给 dict_users[i]。
        all_idxs = list(set(all_idxs) - dict_users[i])
        # 从 all_idxs 中移除已经被当前用户选中的索引，确保每个用户获得的数据是唯一的，实现 I.I.D. 采样。
    return dict_users
    # 在为所有用户分配了数据索引后，返回包含这些索引的字典 dict_users。


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    # 创建MNIST数据集的实例，指定训练集（train=True），如果数据集尚未在本地可用，则下载它（download=True）。
    # 使用transforms.Compose来定义一个转换序列，
    # 其中包括将图像转换为PyTorch张量（transforms.ToTensor()）和进行归一化处理（transforms.Normalize((0.1307,),
    # (0.3081,))），使图像数据的分布具有特定的均值和标准差。
    num = 100
    # 设置一个变量num为100，这个变量表示接下来要分配数据的用户数量。
    d = mnist_noniid(dataset_train, num)
    # 调用之前定义的mnist_noniid函数，传入MNIST训练数据集实例和用户数量100。
    # 这个函数将为这些用户以非独立同分布的方式采样数据，并返回一个包含每个用户数据索引的字典，存储在变量d中。
