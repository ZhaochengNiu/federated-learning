#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    # 定义了一个名为 FedAvg 的函数，它接收一个参数 w，这里 w 应该是一个字典的列表，其中每个字典包含了一个客户端模型的参数。
    w_avg = copy.deepcopy(w[0])
    # 创建一个新的字典 w_avg, 它是列表 w 中第一个字典的深拷贝。这意味着 w_avg 将包含与 w[0] 相同的键和值的副本，但它们是独立的对象。
    for k in w_avg.keys():  # 遍历 w_avg 字典中的所有键（即模型参数的名称）。
        for i in range(1, len(w)):  # 对于 w_avg 中的每个键 k，再次遍历列表 w 中的每个字典，从第二个元素开始（索引为1），直到列表的末尾。
            w_avg[k] += w[i][k]   # 将当前键 k 在列表 w 中第 i 个字典中的值累加到 w_avg[k]。这里假设所有字典中的参数 k 都是相同类型的数值，可以直接进行加法操作。
        w_avg[k] = torch.div(w_avg[k], len(w))  # 在累加完所有客户端的参数之后，将 w_avg[k] 中的值除以列表 w 的长度，即客户端的数量，以计算平均值。
    return w_avg
