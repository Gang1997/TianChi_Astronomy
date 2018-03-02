# Created by [Yuexiong Ding] at 2018/2/11
# 对数据集的索引文件进行进行采样
#

import pandas as pd

# 原始数据集索引文件地址
INDEX_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\train_index.csv'
# 采样后数据集索引文件的地址
SAMPLE_INDEX_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\NormalizedTrainingData\Index'
# 采样次数
M = 1
# 每次采样的样本个数
N = 4000

# 读取索引文件
Data = pd.read_csv(INDEX_FILE_PATH)
SampleData = Data.sample(n=N, weights={'galaxy': 0.25, 'qso': 0.25, 'star': 0.25, 'unknown': 0.25}, axis=0)
print(SampleData)
# for i in range(M):
#     print('第' + str(i + 1) + '次采样\n')
#     SampleData = Data.sample(n=N, weights=[])
#     SamplePath = SAMPLE_INDEX_FILE_PATH + '\\' + str(i + 1) + '.csv'
#     SampleData.to_csv(SamplePath, index=False, sep=',')

print("success")

