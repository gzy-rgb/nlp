import jieba
import re
import pickle
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 将单词去掉停用词后，转换为id
def convert(path_input, path_output):
    len = 12500
    files = []
    data = []
    labels = []
    zeros = [0] * len
    ones = [1] * len
    labels.extend(zeros)
    labels.extend(ones)
    labels.extend(zeros)
    labels.extend(ones)
    word_to_id = pickle.load(open('./data/word_to_id.pkl', 'rb'))
    stoplist = [line.strip() for line in open('./data/stopwords.txt','r',encoding = 'utf-8').readlines()]
    for i in path_input:
        files.extend(os.listdir(i))
    for i in range(0, 4 * len):
        if(i < len):
            path = path_input[0] + '/' + files[i]
        elif(i < 2 * len):
            path = path_input[1] + '/' + files[i]
        elif(i < 3 * len):
            path = path_input[2] + '/' + files[i]
        else:
            path = path_input[3] + '/' + files[i]
        print(i)
        res = []
        for line in open(path, 'r', encoding = 'utf-8'):
            line = line.strip().lower()
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", line) #去掉标点与特殊字符
            words = line.split(' ')
            for word in words:
                if word not in stoplist:
                    res.append(word_to_id.get(word, 0))    # 如果没找到用0补
        data.append(res)
    with open(path_output[0], 'wb') as f:
        pickle.dump((data[0 : 2 * len], labels[0 : 2 * len]), f)
    with open(path_output[1], 'wb') as f:
        pickle.dump((data[2 * len : 4 * len], labels[2 * len : 4 * len]), f)


# 创建dataloader
def build_dataloader(path, pad_size, batch_size):
    with open(path, 'rb') as f:
        data, labels = pickle.load(f)
    for i in range(len(data)):
        if(len(data[i]) > pad_size):          #单词量比所给长度大，截断
            data[i] = data[i][:pad_size]
        else:
            data[i].extend([0] *(pad_size - len(data[i])))  #单词量比所给的小，填充0
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 加载词向量
def load_word_vector(path_val, dim):
    f = open(path_val,'r',encoding = 'utf-8')
    word_to_id = {}
    embeddings = []
    for idx, line in enumerate(f.readlines()):
        lin = line.strip().split(' ')
        word_to_id[lin[0]] = idx
        vector = [float(x) for x in lin[1:dim + 1]]
        embeddings.append(vector)
    embeddings = np.asarray(embeddings, dtype= 'float32')
    f.close()
    with open('./data/word_to_id.pkl', 'wb') as f:
        pickle.dump(word_to_id, f)
    np.save('./data/embeddings.npy', embeddings)

if __name__ == '__main__':
    load_word_vector('./data/glove.6B.300d.txt', 300)
    path_input = ['./data/train/neg', './data/train/pos', './data/test/neg', './data/test/pos']
    path_output = ['./data/train.pkl', './data/test.pkl']
    convert(path_input, path_output)

