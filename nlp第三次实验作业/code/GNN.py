import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
# 使用上一篇博客中用到数据集处理函数
from coraDatasetsProcess import main

node_nums, feature_dims, label_list, feat_Matrix, degree_list, cites, X_Node, X_Neis = main()


# 定义图卷积模型
class gnnModel(torch.nn.Module):
    def __init__(self):
        super(gnnModel, self).__init__()
        self.lin1 = nn.Linear(in_features=1433, out_features=7)

    def forward(self, x, dig_list, A):
        print(A.shape)
        N = len(x)
        I_list = [1 for i in range(N)]
        I = np.diag(I_list)  # 单位矩阵
        A = A + I + I  # 添加自循环
        diags = np.diag(dig_list ** (-0.5))
        pre = np.dot(np.dot(np.dot(diags, A), diags), x)
        pre = pre.astype(np.float32)
        pre = torch.from_numpy(pre)  # 从numpy的ndarray格式转化为tensor张量
        x = self.lin1(pre)
        return x


# 数据预处理
def processData():
    A = np.zeros((node_nums, node_nums))
    for i in range(len(X_Node)):
        source = X_Node[i].item()
        target = X_Neis[i].item()
        A[source][target] = 1  # 构造邻接矩阵
    return A


# 实例化模型并训练
def modelStart(A):
    net = gnnModel()  # 实例化图神经网络模型
    net.train()  # 训练模式

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    out = 0
    lossList = []
    for epoch in range(200):
        optimizer.zero_grad()
        out = net(feat_Matrix[:500], degree_list[:500], A[:500, :500])
        loss = loss_function(out, torch.tensor(label_list[:500], dtype=torch.long))
        lossList.append(loss)
        print("epoch:", epoch, " loss:", loss)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = net(feat_Matrix, degree_list, A)
        max_value, max_index = torch.max(out.data, 1)
        correct = max_index.eq(torch.tensor(label_list, dtype=torch.long)).sum().item()
        print("the accuracy of node classification is:", correct / len(label_list))
    return lossList


# plot画loss曲线
def plotCora(lossList):
    N = len(lossList)
    y = lossList
    x = [i for i in range(N)]
    p = plt.plot(x, y)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("the loss curve for node classification")
    plt.show()


def mainP():
    A = processData()
    lossList = modelStart(A)
    plotCora(lossList)


if __name__ == '__main__':
    mainP()