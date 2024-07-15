#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
# 导入matplotlib库，并设置其后端为'Agg'，这样即使在没有图形界面的服务器上也能生成图像文件。
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
# 导入所需的库：matplotlib.pyplot用于绘图，copy用于复制对象，
# numpy用于数学运算，torchvision中的datasets和transforms用于加载和处理数据，torch是PyTorch深度学习框架。

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
# 从utils模块的sampling中导入数据采样函数。
from utils.options import args_parser
# 从utils模块的options中导入参数解析函数。
from models.Update import LocalUpdate
# 从models模块的Update中导入LocalUpdate类。
from models.Nets import MLP, CNNMnist, CNNCifar
# 从models模块的Nets中导入MLP、CNNMnist和CNNCifar模型。
from models.Fed import FedAvg
# 从models模块的Fed中导入FedAvg联邦平均算法。
from models.test import test_img
# 从models模块的test中导入test_img测试函数。


if __name__ == '__main__':
    # 这是Python的主程序入口点。如果这个脚本作为主程序运行，那么将执行以下代码块。
    # parse args
    args = args_parser()
    # 解析命令行参数，这些参数将控制程序的行为。
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # 根据args.gpu的值和系统是否支持CUDA，设置PyTorch的设备。如果GPU可用并且args.gpu不是-1，则使用GPU；否则使用CPU。
    # load dataset and split users
    # 根据args.dataset的值加载MNIST或CIFAR-10数据集。如果数据集不被识别，则退出程序。
    if args.dataset == 'mnist':
        # 这部分代码负责加载数据集并根据参数将数据分配给不同的用户。
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    # 获取训练数据集中第一个图像的形状，这通常用于确定输入层的大小。
    img_size = dataset_train[0][0].shape
    # build model
    # 根据args.model和args.dataset的值构建不同类型的模型。如果模型不被识别，则退出程序。
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # 打印全局模型的结构。
    net_glob.train()
    # 将全局模型设置为训练模式。
    # copy weights
    # 获取全局模型的当前权重。
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    # 初始化一些用于记录训练过程中损失和准确率的变量。

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        # 如果args.all_clients为真，则表示在每次迭代中使用所有客户端的数据进行训练。
    for iter in range(args.epochs):
        # 进行args.epochs次迭代训练。
        loss_locals = []
        # 如果不使用所有客户端的数据，随机选择一部分客户端参与训练。
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # 对选中的每个客户端进行本地更新。
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # 将更新后的全局权重加载到全局模型中。
        net_glob.load_state_dict(w_glob)

        # print loss
        # 计算并打印平均损失，记录训练损失。
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    # 绘制训练损失曲线并保存图像。
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    # 在测试之前将模型设置为评估模式。
    net_glob.eval()
    # 在训练集和测试集上测试模型的性能，并打印准确率和损失。
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

