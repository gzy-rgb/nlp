# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, embedding):
        self.embeddings = torch.tensor(np.load('./news/embeddings.npy'))
        self.embedding_size = 300
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 10                                           # 类别数


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embeddings, freeze=False)
        self.lstm = nn.LSTM(config.embedding_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[256, 300, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out