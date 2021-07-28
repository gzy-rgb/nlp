# prrprocess the training and test data
import jieba
import re
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# 把类别转换为id
def class_convert(path_class):
    f = open(path_class,'r',encoding = 'utf-8')
    class_to_id = {}
    #id_to_class = {}
    for idx, line in enumerate(f):
        line = line.strip('\n')
        class_to_id[line] = idx
        #id_to_class[idx] = line
    f.close()
    return class_to_id

# 将词向量转换成字典与矩阵形式
def load_word_vector(path_val, dim):
    f = open(path_val,'r',encoding = 'utf-8')
    word_to_id = {}
    embeddings = []
    next(f)                       # 从第二行开始读取
    for idx, line in enumerate(f.readlines()):
        lin = line.strip().split(' ')
        word_to_id[lin[0]] = idx
        vector = [float(x) for x in lin[1:dim + 1]]
        embeddings.append(vector)
    embeddings = np.asarray(embeddings, dtype= 'float32')
    f.close()
    with open('./news/word_to_id.pkl', 'wb') as f:
        pickle.dump(word_to_id, f)
    np.save('./news/embeddings.npy', embeddings)

# 将文本分词，并且将分词后的单词转换成词字典里的id, 与标签一起构成数字化的训练数据集
def split(path_input, path_output):
    data = []
    labels = []
    class_to_id = class_convert('./news/class.txt')
    f = open('./news/word_to_id.pkl', 'rb')
    word_to_id = pickle.load(f)
    f = open(path_input, 'r', encoding='utf-8')
    for line in tqdm(f):              #进度条
        line = line.strip('\n')
        target, content = line.split('\t')
        target = class_to_id[target]
        pattern = re.compile(r'[^\u4e00-\u9fa5]')     #取出中文字符
        content = re.sub(pattern, '', content)
        words = jieba.lcut(content)
        res = []
        for word in words:
            res.append(word_to_id.get(word, 0))
        data.append(res)
        labels.append(target)
    f.close()
    with open(path_output, 'wb') as f:
        pickle.dump((data, labels), f)

# 短的填充，长的截断
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

if __name__ == '__main__':
    #load_word_vector('./news/sgns.wiki.word', 300)

    split(path_input= './news/cnews.val.txt', path_output= './news/val.pkl')
    split(path_input= './news/cnews.test.txt', path_output= './news/test.pkl')
    split(path_input= './news/cnews.train.txt', path_output= './news/train.pkl')








