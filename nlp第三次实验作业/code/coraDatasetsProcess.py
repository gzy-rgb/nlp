import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
node_num, feat_dim, stat_dim, num_class, T
feat_Matrix, X_Node, X_Neis, dg_list
cora.content共有2708行，每一行代表一个样本点，即一篇论文。
如下所示，每一行由三部分组成，分别是论文的编号，如31336；论文的词向量，一个有1433位的二进制；
论文的类别，如Neural_Networks。
'''


# 处理数据
def processCora(content_path, cite_path):
    # 读取节点属性信息
    with open(content_path, "r") as fp:
        contents = fp.readlines()
    # 读取节点间的边引用关系
    with open(cite_path, "r") as fp:
        cites = fp.readlines()
    # contens为（2708，1435） 表示2708个节点，每个节点为1435维
    # 第一个维度是编号，中间1433维是节点的0 1特征表示向量，最后一位是label分类
    contents = np.array([np.array(l.strip().split("\t")) for l in contents])
    # 取得文章(节点)编号 (2708,1)维，特征表示向量(2708,1433)维，文章标签(2708,1)
    paper_list, feat_list, label_list = np.split(contents, [1, -1], axis=1)
    # 序列化操作，维度转换 如paper_list变为(1,2708)维
    paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)
    # 将文章编号加上索引映射成字典的形式（由0到2707）
    paper_dict = dict([(int(key), float(val)) for val, key in enumerate(paper_list)])
    # 将八个分类标签存储在列表中
    labels = list(set(label_list))
    # 标签也转化为字典形式存储 {”标签“:索引}
    label_dict = dict([(key, val) for val, key in enumerate(labels)])
    """
    现在关于论文引用网络的节点属性信息，
    包括节点编号的映射，节点的特征表示，节点的标签都已经提取完毕。
    接下来处理论文引用网络的边引用关系
    """
    # 将论文引用关系存储在列表中 [[论文编号，论文编号],[论文编号，论文编号]..........]
    cites = [i.strip().split("\t") for i in cites]  # 共5429条边
    # 因为元数据及中论文编号过大，直接存储边的话邻接矩阵开销太大
    # 这里利用编号与索引的字典映射转化为存储索引的边信息，减小开销
    cite = []
    for i in cites:
        key = int(i[0])
        value = int(i[1])
        # 根据索引存储边的信息
        cite.append([paper_dict[key], paper_dict[value]])
    cites = np.array(cite, np.int64).T  # 维度为(2,5429)，边的引用关系，可直接被geometric所使用
    # 本来有5429条边，转化成无向图，边数加倍 （2,10858）维度
    cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)
    # 计算每一个节点的度 _是编号索引列表，degree_list存储节点对应的度
    _, degree_list = np.unique(cites[0, :], return_counts=True)

    """
    一些输入数据
    """
    # 节点数量
    node_nums = len(paper_list)
    # 特征维度
    feature_dims = feat_list.shape[1]
    # 标签
    num_class = len(label_list)
    # 特征表示矩阵转化为tensor张量格式
    feat_Matrix = torch.Tensor(feat_list.astype(np.float32))
    # 将原始的(2,10858)维度的边矩阵进行切分，分成两个，分别是源节点和目标节点矩阵
    # 两个维度都是(1,10858)
    X_Node, X_Neis = np.split(cites, 2, axis=0)
    # print(cites)
    # print(X_Node)
    # print(X_Neis)
    # 将[[......]]格式的X_node和X_Neis转化为tensor张量格式
    X_Node, X_Neis = torch.from_numpy(np.squeeze(X_Node)), \
                     torch.from_numpy(np.squeeze(X_Neis))

    dg_list = degree_list[X_Node]
    # 标签转化为纯列表形式存储
    label_list = np.array([label_dict[i] for i in label_list])
    # 转化为张量格式
    label_list = torch.from_numpy(label_list)
    return node_nums, feature_dims, label_list, feat_Matrix, degree_list, cites, X_Node, X_Neis


# 打印cora数据集的各种属性信息
def printCora(node_nums, feature_dims, label_list, feat_Matrix, degree_list, cites, X_Node, X_Neis):
    print("{}Data Process Info{}".format("*" * 20, "*" * 20))
    print("==> Number of node : {}".format(node_nums))
    print("==> Number of edges : {}/2={}".format(cites.shape[1], int(cites.shape[1] / 2)))
    print("==> Number of classes : {}".format(len(np.unique(label_list))))
    print("==> Dimension of node features : {}".format(feature_dims))
    print("==> Shape of feat_Matrix : {}".format(feat_Matrix.shape))
    print("==> Shape of X_Node : {}".format(X_Node.shape))
    print("==> Shape of X_Neis : {}".format(X_Neis.shape))
    print("==> Length of dgree_list : {}".format(len(degree_list)))


# main()函数
def main():
    # 数据集路径
    content_path = "data/cora/cora.content"  # 节点属性信息
    cite_path = "data/cora/cora.cites"  # 论文引用信息
    """
    以下分别是节点个数、节点特征维度、节点标签、特征表示矩阵、度矩阵、
    边(2,2*edges)、源节点、目标节点
    """
    node_nums, feature_dims, label_list, feat_Matrix, degree_list, cites, X_Node, X_Neis \
        = processCora(content_path, cite_path)
    printCora(node_nums, feature_dims, label_list, feat_Matrix, degree_list, cites, X_Node, X_Neis)
    return node_nums, feature_dims, label_list, feat_Matrix, degree_list, cites, X_Node, X_Neis


if __name__ == '__main__':
    main()