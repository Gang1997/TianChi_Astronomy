# Created by [Yuexiong Ding] at 2018/2/12
# 测试数据降维处理
#
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

# 测试数据文件存放目录
TEST_RAW_DATA_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\TestData'
# 降维后数据保存的地址
TEST_NORMAL_DATA_ROOT_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\NormalizedTrainingData\Samples\TestData'
# 数据维数（降维时使用）
DIMENSION = 1024


for (root, dirs, files) in os.walk(TEST_RAW_DATA_PATH):
    for filename in files:
        path = os.path.join(root, filename)

        print()
#
# # 定义类别字典
# TYPE_DICTIONARY = {'galaxy': np.array([1, 0, 0, 0]), 'qso': np.array([0, 1, 0, 0]), 'star': np.array([0, 0, 1, 0]),
#                    'unknown': np.array([0, 0, 0, 1])}
#
# SampleIndex = pd.read_csv(SAMPLE_INDEX_FILE_PATH + '\\' + str(SAMPLE_NO) + '.csv')
# SampleId = SampleIndex['id']
# SampleType = SampleIndex['type']
#
# # 对类别进行编码,并保存txt文件 [galaxy  qso  star  unknown]
# # 1、调用 pd.get_dummies，不全适用，采样的index文件中包含的类别不一定相同
# # SampleTypeEncode = np.array(pd.get_dummies(SampleType)).reshape(1024, 4)
# # 2、人工编码
# SampleTypeEncode = np.array([TYPE_DICTIONARY[SampleType[0]]])
# for j in range(1, SampleType.size):
#     SampleTypeEncode = np.r_[SampleTypeEncode, np.array([TYPE_DICTIONARY[SampleType[j]]])]
# np.savetxt(SAMPLE_ENCODE_TYPE_FILE_PATH + '\\' + str(SAMPLE_NO) + '.txt', SampleTypeEncode, delimiter=',')
# print('编码完成!')
#
# # 读取采样中的每个样本，拼成样本矩阵
# SampleData = np.array([np.loadtxt(SAMPLE_DATA_FILE_PATH + '\\' + str(SampleId[0]) + '.txt')])
# for k in range(1, SampleId.size):
#     print('拼接第' + str(k) + '个样本数据')
#     SampleData = np.r_[SampleData, np.array([np.loadtxt(SAMPLE_DATA_FILE_PATH + '\\' + str(SampleId[k]) + '.txt')])]
# print('拼成的矩阵维数：' + str(np.shape(SampleData)))
#
# # PCA降维
# pca = PCA(n_components=DIMENSION)
# SampleData_PCA = pca.fit_transform(SampleData)
# print(pca.explained_variance_ratio_)
# print('降维后矩阵维数：' + str(np.shape(SampleData_PCA)))
# # 保存降维数据到txt文件
# np.savetxt(SAMPLE_PCA_DATA_FILE_PATH + '\\' + str(DIMENSION) + 'D\\TrainingData.txt',
#            SampleData_PCA[:int(SampleId.size * SPLIT_PERCENT), :],
#            delimiter=',')
# np.savetxt(SAMPLE_PCA_DATA_FILE_PATH + '\\' + str(DIMENSION) + 'D\\DevData.txt',
#            SampleData_PCA[int(SampleId.size * SPLIT_PERCENT):, :],
#            delimiter=',')
# print('数据降维完毕！')
#
# # 读取数据，并进行降维处理
# # for i in range(M):
# #     SampleIndex = pd.read_csv(SAMPLE_INDEX_FILE_PATH + '\\' + str(i + 1) + '.csv')
# #     SampleId = SampleIndex['id']
# #     SampleType = SampleIndex['type']
# #
# #     # 对类别进行编码,并保存txt文件 [galaxy  qso  star  unknown]
# #     # 1、调用 pd.get_dummies，不全适用，采样的index文件中包含的类别不一定相同
# #     # SampleTypeEncode = np.array(pd.get_dummies(SampleType)).reshape(1024, 4)
# #     # 2、人工编码
# #     SampleTypeEncode = np.array([TYPE_DICTIONARY[SampleType[0]]])
# #     for j in range(1, SampleType.size):
# #         SampleTypeEncode = np.r_[SampleTypeEncode, np.array([TYPE_DICTIONARY[SampleType[j]]])]
# #     np.savetxt(SAMPLE_ENCODE_TYPE_FILE_PATH + '\\' + str(i + 1) + '.txt', SampleTypeEncode, delimiter=',')
# #     print('第' + str(i + 1) + '次编码完成')
# #
# #     # 读取每个样本，拼成样本矩阵
# #     SampleData = np.array([np.loadtxt(SAMPLE_DATA_FILE_PATH + '\\' + str(SampleId[0]) + '.txt')])
# #     for k in range(1, SampleId.size):
# #         print('拼接第' + str(k) + '个样本数据')
# #         SampleData = np.r_[SampleData, np.array([np.loadtxt(SAMPLE_DATA_FILE_PATH + '\\' + str(SampleId[k]) + '.txt')])]
# #     print('拼成的矩阵维数：' + str(np.shape(SampleData)))
# #
# #     # PCA降维
# #     pca = PCA(n_components=D)
# #     SampleData_PCA = pca.fit_transform(SampleData)
# #     print(pca.explained_variance_ratio_)
# #     print('降维后矩阵维数：' + str(np.shape(SampleData_PCA)))
# #     # 保存降维数据到txt文件
# #     np.savetxt(SAMPLE_PCA_DATA_FILE_PATH + '\\' + str(D) + 'D\\' + str(i + 1) + '.txt', SampleData_PCA, delimiter=',')
# #
# # print('数据降维完毕！')
