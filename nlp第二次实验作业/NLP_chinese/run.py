import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch
from tqdm import tqdm
import numpy as np
from preprocess import class_convert
from preprocess import build_dataloader
from models.RNN import Model
from models.RNN import Config

def Train(model, device, train, optimizer, epochs, log_interval):
    model.train()
    sum = 0
    for batch_idx, (data, target) in enumerate(train):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        #print(batch_idx)
        data, target = data.to(device), target.to(device)  # data之指的是一个batch的tensor
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        sum = sum + loss.item()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * len(data), len(train.dataset),
                        100. * batch_idx / len(train), loss.item()))

    sum = sum / (batch_idx + 1)
    return sum


def Test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros((10, 10), dtype=torch.int32).to(device) #混淆矩阵
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(0, len(target)):
                confusion_matrix[target[i]][pred[i]] += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(confusion_matrix.cpu())


# 参数设置
learning_rate = 0.01
batch_size = 128
pad_size = 200
epochs = 10
log_interval = 1



word_to_id = pickle.load(open('./news/word_to_id.pkl', 'rb'))        # 词典
embeddings = np.load('./news/embeddings.npy')  #词向量矩阵

# dataloader
train = build_dataloader('./news/train.pkl', pad_size = pad_size, batch_size = batch_size)
val = build_dataloader('./news/val.pkl', pad_size = pad_size, batch_size = batch_size)
test = build_dataloader('./news/test.pkl', pad_size = pad_size, batch_size = batch_size)

# 建立模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = Config(embeddings)
model = Model(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

# 训练
for epoch in range(1, epochs + 1):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    Train(model, device, train, optimizer, epoch, log_interval)
    Test(model, device, test)


print()

