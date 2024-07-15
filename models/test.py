#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    # 定义了 test_img 函数，它接收三个参数：
    # net_g：要测试的神经网络模型。
    # datatest：包含测试数据的数据集。
    # args：包含测试配置的参数，例如批量大小（bs）、是否使用 GPU（gpu）、是否打印详细信息（verbose）等。
    net_g.eval()  # 将模型 net_g 设置为评估模式，这会关闭模型中的 Dropout 和 Batch Normalization 层的训练行为。
    # testing
    test_loss = 0  # 初始化测试过程中累积的损失和为 0。
    correct = 0  # 初始化测试过程中正确预测的数量为 0。
    data_loader = DataLoader(datatest, batch_size=args.bs)
    # 创建一个 DataLoader 对象，用于从 datatest 加载数据，批量大小由 args.bs 指定。
    l = len(data_loader)
    # 获取测试数据加载器中的批次数量。
    for idx, (data, target) in enumerate(data_loader):  # 遍历测试数据加载器中的所有批次。
        if args.gpu != -1:
        # 如果 args.gpu 不等于 -1，表示使用 GPU，将数据和目标转移到 GPU。
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # 执行模型的前向传播，计算输入数据 data 的对数概率。
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # 计算交叉熵损失，使用 reduction='sum' 将批次内的损失求和，然后累加到 test_loss。
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # 获取最大概率所对应的预测类别。
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        # 计算预测正确的样本数量。
    test_loss /= len(data_loader.dataset)
    # 计算整个测试集上的平均损失。
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # 计算测试集上的准确率。
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    # 如果 args.verbose 为真，则打印测试集上的平均损失和准确率。
    return accuracy, test_loss
    # 返回测试集上的准确率和平均损失。

