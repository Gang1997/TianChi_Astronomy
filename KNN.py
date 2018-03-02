
# Created by [Yuexiong Ding] at 2018/3/1
# KNN分
#

from sklearn import neighbors
from sklearn import metrics
from sklearn import decomposition
import numpy as np
import pandas as pd

Test_DATA_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\TestData'
Test_INDEX_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\test_index.csv'

CLASSES = ['galaxy', 'qso', 'star', 'unknown']

# 加载索引数据
Index = pd.read_csv(Test_INDEX_FILE_PATH)['id']

# 最终分类结果
Result = []

# # Type = pd.read_csv('./DataSet/Normalized/TestIndex2.csv')['type']
trX = np.loadtxt('./DataSet/Normalized/TrainingData.txt', delimiter=',')
# trX = np.loadtxt('./DataSet/Normalized/TestData.txt', delimiter=',')
trY = np.argmax(np.loadtxt('./DataSet/Normalized/TrainingLabels.txt', delimiter=','), axis=1)
# trY = np.argmax(np.loadtxt('./DataSet/Normalized/TestLabels.txt', delimiter=','), axis=1)

# teY = np.argmax(np.loadtxt('./DataSet/Normalized/DevLabels.txt', delimiter=','), axis=1)
#
# pca = decomposition.PCA(n_components=100)
pca = decomposition.KernelPCA(kernel='cosine', n_components=100)
# pca.fit(trX)
trXPCA = pca.fit_transform(trX)
# print('数据降维完毕,维数 %d 维' % len(teXPCA[0]))

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(trXPCA, trY)
# predict = []
# for i in range(len(trX)):

for i in range(len(Index)):
    teX = np.array([np.loadtxt(Test_DATA_FILE_PATH + '\\' + str(Index[i]) + '.txt', delimiter=',')])
    teXPCA = pca.transform(teX)
    predict = knn.predict(teXPCA)
    print('预测第 %d 条, 预测结果 %s' % (i, CLASSES[predict[0]]))
    Result.append([Index[i], CLASSES[predict[0]]])

df = pd.DataFrame(Result, columns=['key', 'predicted class'])
df.to_csv('./DataSet/Predict/Result_KNN_PCA.csv', index=False)

# print("micro recall accuracy %g" % metrics.recall_score(teY, predict, average='micro'))
# print("macro recall accuracy %g" % metrics.recall_score(teY, predict, average='macro'))
# print("confusion_matrix:")
# print(metrics.confusion_matrix(teY, predict))
#
# print("classification_report:")
# print(metrics.classification_report(teY, predict, target_names=CLASSES))
# print("f1 accuracy %g" % metrics.f1_score(teY, predict, average='weighted'))







