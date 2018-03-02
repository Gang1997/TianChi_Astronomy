# Created by [Yuexiong Ding] at 2018/2/12
# 归一化数据
#

import pandas as pd
import numpy as np

# 原始数据集索引文件地址
INDEX_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\train_index.csv'
# 原始数据集文件存放地址
DATA_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\TrainingData'
# 采样数据文件存放根目录
SAMPLE_ROOT_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\NormalizedTrainingData'
# 采样后数据集索引文件的地址
SAMPLE_INDEX_FILE_PATH = SAMPLE_ROOT_PATH + r'\Index'
# 采样后数据集存放地址
SAMPLE_DATA_FILE_PATH = SAMPLE_ROOT_PATH + r'\Samples\TrainingData'
# 均值
MEAN = 0
# 方差
VARIANCE = 0

# 读取索引文件
IndexData = pd.read_csv(INDEX_FILE_PATH)
IndexDataId = IndexData['id']

# 计算均值向量
for i in range(IndexDataId.size):
    print('求均值：第' + str(i + 1) + '样本')
    # i 从 0 开始
    MEAN = (i * MEAN + np.loadtxt(DATA_FILE_PATH + '\\' + str(IndexDataId[i]) + '.txt', delimiter=',')) / (i + 1)
    # 打印出均值向量
    print('均值向量为：' + str(MEAN))
# 将均值写入文件保存起来
MeanFrame = pd.DataFrame({'Mean': MEAN})
MeanFrame.to_csv(SAMPLE_ROOT_PATH + '\\' + 'Mean.csv', index=False, sep=',')

# 计算方差
for i in range(IndexDataId.size):
    print('求方差：第' + str(i + 1) + '样本')
    VARIANCE = (i * VARIANCE + (
            np.loadtxt(DATA_FILE_PATH + '\\' + str(IndexDataId[i]) + '.txt', delimiter=',') - MEAN) ** 2) / (i + 1)
    # 打印出方差向量
    print('方差向量为：' + str(VARIANCE))
# 将方差写入文件保存起来
VarianceFrame = pd.DataFrame({'Variance': VARIANCE})
VarianceFrame.to_csv(SAMPLE_ROOT_PATH + '\\' + 'Variance.csv', index=False, sep=',')

# 归一化采样的样本
for i in range(IndexDataId.size):
    print('归一化：第' + str(i + 1) + '样本')
    NormalData = (np.loadtxt(DATA_FILE_PATH + '\\' + str(IndexDataId[i]) + '.txt', delimiter=',') - MEAN) / VARIANCE
    np.savetxt(SAMPLE_DATA_FILE_PATH + '\\' + str(IndexDataId[i]) + '.txt', NormalData, delimiter=',')
    # 打印出归一化向量
    print('归一化向量为：' + str(NormalData))
