#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    # --epochs 表示训练的轮数。
    # --num_users 表示参与联邦学习的用户的总数。
    # --frac 表示每轮参与训练的用户比例。
    # --local_ep 表示用户在本地训练的轮数。
    # --local_bs 表示用户本地训练的批处理大小。
    # --bs 表示测试时的批处理大小。
    # --lr 表示学习率。
    # --momentum 表示SGD动量的值。
    # --split 表示训练-测试拆分的类型。
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    # --model 表示模型的名称。
    # --kernel_num 表示每种类型的卷积核数量。
    # --kernel_sizes 表示卷积核的大小。
    # --norm 表示使用的归一化类型。
    # --num_filters 表示卷积网络的过滤器数量。
    # --max_pool 表示是否使用最大池化。
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # --dataset 表示数据集的名称。
    # --iid 表示是否使用独立同分布数据。
    # --num_classes 表示类别的总数。
    # --num_channels 表示图像的通道数。
    # --gpu 表示使用的GPU编号，-1 表示使用CPU。
    # --stopping_rounds 表示提前停止的训练轮数。
    # --verbose 表示是否打印详细信息。
    # --seed 表示随机种子。
    # --all_clients 表示是否对所有客户端进行聚合。
    args = parser.parse_args()
    return args
