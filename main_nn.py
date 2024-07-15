#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
# 导入 matplotlib.pyplot 模块，用于绘图。
import matplotlib.pyplot as plt


import torch
# 导入 PyTorch 库，用于构建和训练神经网络。
import torch.nn.functional as F
# 从 PyTorch 库中导入 nn.functional 模块，包含神经网络的函数式接口。
from torch.utils.data import DataLoader
# 从 PyTorch 库中导入 DataLoader 类，用于加载数据集。
import torch.optim as optim
# 从 PyTorch 库中导入 optim 模块，包含优化器。
from torchvision import datasets, transforms
# 从 torchvision 库中导入 datasets 和 transforms 模块，用于处理图像数据集和定义图像转换。

from utils.options import args_parser
# 从 utils 模块中导入 args_parser 函数，用于解析命令行参数。
from models.Nets import MLP, CNNMnist, CNNCifar
# 从 models 模块的 Nets 中导入定义的神经网络模型 MLP, CNNMnist, CNNCifar。


def test(net_g, data_loader):
    # testing
    # 定义测试函数，net_g 是要测试的网络模型，data_loader 是包含测试数据的加载器。
    net_g.eval()  # 将网络设置为评估模式，这会关闭Dropout和Batch Normalization层的训练行为。
    test_loss = 0 # 初始化测试损失为0。
    correct = 0  # 初始化正确的预测数为0。
    l = len(data_loader)   # 获取数据加载器中的批次数量。
    for idx, (data, target) in enumerate(data_loader):  # 遍历数据加载器中的所有批次。
        data, target = data.to(args.device), target.to(args.device)   # 将数据和目标转移到指定的设备（CPU或GPU）。
        log_probs = net_g(data)    # 前向传播，获取模型输出的对数概率。
        test_loss += F.cross_entropy(log_probs, target).item()  # 计算交叉熵损失，并累加到测试损失。
        y_pred = log_probs.data.max(1, keepdim=True)[1]   # 找到最大概率对应的类别。
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()  # 计算预测正确的数量。

    test_loss /= len(data_loader.dataset)  # 计算预测正确的数量。
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))    # 返回测试集的正确预测数和平均损失。

    return correct, test_loss


if __name__ == '__main__':
    # parse args
    # 调用 args_parser() 函数来解析命令行参数，这些参数通常用于控制脚本的行为，例如选择数据集、模型类型、学习率等。
    args = args_parser()
    # 根据 args.gpu 的值和系统是否支持 CUDA，设置 PyTorch 的设备。
    # 如果 GPU 可用并且 args.gpu 不是 -1（通常表示不使用 GPU），则使用 GPU；否则使用 CPU。
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # 设置随机种子，以确保实验的可重复性。
    torch.manual_seed(args.seed)

    # load dataset and split users
    # 这一部分代码是条件判断，根据 args.dataset 的值来加载不同的数据集。
    # 根据参数 args.dataset 的值，选择加载 MNIST 或 CIFAR-10 数据集，并应用适当的数据转换。
    # 如果参数不是这两个值之一，脚本将退出并显示错误信息。
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        img_size = dataset_train[0][0].shape  # 获取第一个训练数据样本的形状，这通常用于确定输入层的大小。
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    # 根据参数 args.model 和 args.dataset 的组合，实例化不同的模型类，并将其移动到指定的设备上。
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # 打印模型的摘要，以便于查看模型结构。
    print(net_glob)

    # training
    # 创建一个 SGD 优化器，使用从 args 解析出的学习率和动量参数。
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    # 创建一个 DataLoader 对象，用于从 dataset_train 加载数据，批大小设置为 64，并且每个 epoch 开始时都会打乱数据顺序。
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    list_loss = []  # 初始化一个列表来记录每个epoch的平均损失。
    net_glob.train()   # 将网络设置为训练模式。
    for epoch in range(args.epochs):  # 进行指定次数的迭代（epochs）。
        batch_loss = []     # 初始化一个列表来记录当前epoch中每个批次的损失。
        for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练数据加载器。
            data, target = data.to(args.device), target.to(args.device)    # 将数据和目标转移到指定的设备。
            optimizer.zero_grad()   # 清除之前的梯度，为新的批次准备梯度计算。
            output = net_glob(data)   # 前向传播，获取模型的输出。
            loss = F.cross_entropy(output, target)  # 前向传播，获取模型的输出。
            loss.backward()   # 反向传播，计算梯度。
            optimizer.step()  # 更新模型参数。
            if batch_idx % 50 == 0: # 每50个批次打印一次训练进度。
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))  # 记录当前批次的损失。
            batch_loss.append(loss.item())   # 将当前epoch的平均损失添加到损失列表中。
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # plot loss
    # 创建一个新的 matplotlib 图。
    plt.figure()
    # 绘制训练损失的图表。range(len(list_loss)) 生成一个从 0 到 list_loss 长度减 1 的序列，
    # 代表每个 epoch 的索引；list_loss 是记录每个 epoch 结束时损失的列表。
    plt.plot(range(len(list_loss)), list_loss)
    # 设置 x 轴的标签为 "epochs"。
    plt.xlabel('epochs')
    # 设置 y 轴的标签为 "train loss"。
    plt.ylabel('train loss')
    # 将损失图表保存为 PNG 图片文件。文件名基于数据集、模型和 epoch 数量动态生成，
    # 保存在 ./log/ 目录下。如果 ./log/ 目录不存在，matplotlib 会抛出 FileNotFoundError。
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    # 这部分代码是条件判断，根据 args.dataset 的值来加载测试数据集。
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
