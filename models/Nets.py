#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


# 定义了一个名为 MLP 的类，它继承自 torch.nn.Module，这是所有神经网络模块的基类。
class MLP(nn.Module):
    # 定义了 MLP 类的构造函数，它接收三个参数：
    # dim_in：输入层的维度，即输入特征的数量。
    # dim_hidden：隐藏层的维度。
    # dim_out：输出层的维度，通常是分类任务中类别的数量。
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()  # 调用父类 nn.Module 的构造函数，这是 Python 中类的初始化的惯用方法。
        self.layer_input = nn.Linear(dim_in, dim_hidden)  # 创建一个线性层（nn.Linear），它将输入特征从 dim_in 维度转换到 dim_hidden 维度。
        self.relu = nn.ReLU()  # 创建一个 ReLU（Rectified Linear Unit）激活函数层，用于在前向传播过程中引入非线性。
        self.dropout = nn.Dropout()  # 创建一个 Dropout 层，用于在训练过程中随机丢弃（置零）一部分神经元的输出，以减少过拟合。
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)  # 创建第二个线性层，将隐藏层的特征从 dim_hidden 维度转换到输出层的 dim_out 维度。

    def forward(self, x):
        # 定义了 MLP 类的 forward 方法，它是每个 nn.Module 子类必须实现的方法，用于指定如何计算前向传播。
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # 将输入 x 重塑（flatten）为一个二维张量。-1 表示自动计算该维度的大小，使得元素总数保持不变。
        # 这里假设 x 是一个多维张量，通常在图像处理中，x 可能是一个四维张量（batch size, channels, height, width）。
        x = self.layer_input(x)  # 在输入层的输出上应用 Dropout。
        x = self.dropout(x)  # 在输入层的输出上应用 Dropout。
        x = self.relu(x)  # 应用 ReLU 激活函数。
        x = self.layer_hidden(x)  # 将 ReLU 激活后的输出传递给隐藏层
        return x


class CNNMnist(nn.Module):
    # 定义了一个名为 CNNMnist 的类，用于构建针对 MNIST 数据集的卷积神经网络。
    def __init__(self, args):
        # 构造函数接收一个参数 args，它通常是一个包含模型配置的命名空间或对象。
        super(CNNMnist, self).__init__()  # 调用父类 nn.Module 的构造函数。
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        # 定义第一个卷积层 conv1，它使用 args.num_channels 作为输入通道数，输出通道数为 10，卷积核大小为 5x5。
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义第二个卷积层 conv2，输入通道数为第一个卷积层的输出通道数（10），输出通道数为 20，卷积核大小同样为 5x5。
        self.conv2_drop = nn.Dropout2d()
        # 定义一个二维 Dropout 层 conv2_drop，用于在训练过程中随机丢弃卷积层的输出，减少过拟合。
        self.fc1 = nn.Linear(320, 50)
        # 定义第一个全连接层 fc1，输入特征数量为 320（这个数字通常是基于输入图像大小和前两个卷积层的输出维度计算得出的），
        # 输出特征数量为 50。
        self.fc2 = nn.Linear(50, args.num_classes)
        # 定义第二个全连接层 fc2，输入特征数量为第一个全连接层的输出特征数量（50），
        # 输出特征数量为 args.num_classes，即分类任务中的类别数。
    def forward(self, x):
        # 定义 forward 方法，用于指定如何计算前向传播。
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 应用第一个卷积层 conv1，然后使用 ReLU 激活函数和最大池化（pool size 为 2）。
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 应用第二个卷积层 conv2，然后通过 Dropout 层 conv2_drop，接着使用 ReLU 激活函数和最大池化。
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        # 将卷积层和池化层的输出扁平化为一维张量，以匹配全连接层的输入要求。
        x = F.relu(self.fc1(x))
        # 应用第一个全连接层 fc1，然后使用 ReLU 激活函数。
        x = F.dropout(x, training=self.training)
        # 应用 Dropout，training=self.training 参数指示 Dropout 是否处于训练模式。
        x = self.fc2(x)
        # 应用第二个全连接层 fc2，得到最终的分类结果。
        return x


class CNNCifar(nn.Module):
    # 定义了一个名为 CNNCifar 的类，用于构建针对 CIFAR-10 数据集的卷积神经网络。
    def __init__(self, args):
        # 构造函数接收一个参数 args，它通常是一个包含模型配置的命名空间或对象。
        super(CNNCifar, self).__init__()  # 调用父类 nn.Module 的构造函数。
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义第一个卷积层 conv1，输入通道数为 3（CIFAR-10 图像是 RGB 三通道），输出通道数为 6，卷积核大小为 5x5。
        self.pool = nn.MaxPool2d(2, 2)
        # 定义一个最大池化层 pool，池化窗口大小为 2x2，步长为 2。
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第二个卷积层 conv2，输入通道数为第一个卷积层的输出通道数（6），输出通道数为 16，卷积核大小同样为 5x5。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第一个全连接层 fc1，输入特征数量为第二个卷积层输出的扁平化后的维度（16 * 5 * 5），输出特征数量为 120。
        self.fc2 = nn.Linear(120, 84)
        # 定义第二个全连接层 fc2，输入特征数量为第一个全连接层的输出特征数量（120），输出特征数量为 84。
        self.fc3 = nn.Linear(84, args.num_classes)
        # 定义第三个全连接层 fc3，输入特征数量为第二个全连接层的输出特征数量（84），
        # 输出特征数量为 args.num_classes，即 CIFAR-10 数据集中的类别数。

    def forward(self, x):
        # 定义 forward 方法，用于指定如何计算前向传播。
        x = self.pool(F.relu(self.conv1(x)))
        # 应用第一个卷积层 conv1，然后使用 ReLU 激活函数，接着应用最大池化。
        x = self.pool(F.relu(self.conv2(x)))
        # 应用第二个卷积层 conv2，然后使用 ReLU 激活函数，接着应用最大池化。
        x = x.view(-1, 16 * 5 * 5)
        # 将卷积层和池化层的输出扁平化为一维张量，以匹配第一个全连接层的输入要求。
        x = F.relu(self.fc1(x))
        # 应用第一个全连接层 fc1，然后使用 ReLU 激活函数。
        x = F.relu(self.fc2(x))
        # 应用第二个全连接层 fc2，然后使用 ReLU 激活函数。
        x = self.fc3(x)
        # 应用第三个全连接层 fc3，得到最终的分类结果。
        return x
