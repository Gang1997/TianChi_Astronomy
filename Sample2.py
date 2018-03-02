# Created by [Yuexiong Ding] at 2018/2/11
# 对数据集的索引文件进行进行采样
# 过采样和欠采样
import numpy as np
import pandas as pd
import random
import os

# 原始数据集索引文件地址
INDEX_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\train_index.csv'
# 采样后数据集索引文件的地址
# SAMPLE_INDEX_FILE_PATH = './DataSet/Normalized/TrainingIndex2.csv'
# SAMPLE_INDEX_FILE_PATH = './DataSet/Normalized/DevIndex.csv'
SAMPLE_INDEX_FILE_PATH = './DataSet/Normalized/TestIndex3.csv'
# SAMPLE_LABEL_FILE_PATH = './DataSet/Normalized/TrainingLabels2.txt'
# SAMPLE_LABEL_FILE_PATH = './DataSet/Normalized/DevLabels.txt'
SAMPLE_LABEL_FILE_PATH = './DataSet/Normalized/TestLabels3.txt'
# 定义类别字典
TYPE_DICTIONARY = {'galaxy': [1, 0, 0, 0], 'qso': [0, 1, 0, 0], 'star': [0, 0, 1, 0], 'unknown': [0.4/3, 0.4/3, 0.4/3, 0.6]}

# 读取索引文件
Data = pd.read_csv(INDEX_FILE_PATH)

Temp = []
Labels = []
for i in range(len(Data)):
    if len(Labels) == 10000:
        break
    if (Data.iloc[[i]]['type'] == 'star').bool():
        if random.randint(1, 10000) > 100:
            continue
    elif (Data.iloc[[i]]['type'] == 'unknown').bool():
        if random.randint(1, 100) > 10:
            continue
    elif (Data.iloc[[i]]['type'] == 'galaxy').bool():
        if random.randint(1, 100) > 50:
            continue
    print(i)
    Labels.append(TYPE_DICTIONARY[Data.iloc[i]['type']])
    Temp.append([Data.iloc[i]['id'], Data.iloc[i]['type']])

# 保存
NewIndex = pd.DataFrame(Temp, columns=['id', 'type'])
NewLabels = np.array(Labels)
NewIndex.to_csv(SAMPLE_INDEX_FILE_PATH, index=False, sep=',')
np.savetxt(SAMPLE_LABEL_FILE_PATH, NewLabels, delimiter=',')

print("success")

