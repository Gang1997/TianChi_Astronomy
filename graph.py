# Created by [Yuexiong Ding] at 2018/3/1
# 散点图
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition

Type = pd.read_csv('./DataSet/Normalized/TestIndex2.csv')['type']
Data = np.loadtxt('./DataSet/Normalized/TestData2.txt', delimiter=',')
# pca = decomposition.PCA(n_components=3)
kpca = decomposition.KernelPCA(kernel='cosine', n_components=100)
# DataPCA = pca.fit_transform(Data)
DataPCA = kpca.fit_transform(Data)
# DataPCA = np.array(pca.fit_transform(Data)).sum(axis=1)
print('数据降维完毕,维数 %d 维' % len(DataPCA[0]))

for i in range(len(Type)):
    x = range(len(DataPCA[0]))
    y = DataPCA[i]
    if Type[i] == 'galaxy':
        plt.scatter(x, y, c='r')
    if Type[i] == 'qso':
        plt.scatter(x, y, c='y')
    # if Type[i] == 'star':
    #     plt.plot(x, y, c='pink')
    #     plt.plot(x, y, c='pink')
    # if Type[i] == 'unknown':
    #     plt.plot(x, y, c='g')
    print('绘制第 %d 条' % i)

plt.show()
