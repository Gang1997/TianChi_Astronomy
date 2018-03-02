# Created by [Yuexiong Ding] at 2018/2/28
# 预测
#
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# CKPT_PATH = './Ckpt_Dir/adam2'
CKPT_PATH = './Ckpt_Dir/GradientDescent_PCA_100_1024_512_128_4'
Test_DATA_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\TestData'
Test_INDEX_FILE_PATH = r'D:\MyProjects\TianChi\Astronomy\DataSet\RawData\test_index.csv'

KEEP_INPUT = 1
KEEP_HIDDEN = 1

# 类别字典
CLASSES = ['galaxy', 'qso', 'star', 'unknown']

# 最终分类结果
Result = []

# 加载索引数据
Index = pd.read_csv(Test_INDEX_FILE_PATH)['id']

fix_X = np.loadtxt('./DataSet/Normalized/TrainingData.txt', delimiter=',')
pca = PCA(n_components=100)
pca.fit(fix_X)

# X = tf.placeholder('float', [None, 2600])
X = tf.placeholder('float', [None, 100])
# X = tf.placeholder('float', [None, 5])
# Y = tf.placeholder('float', [None, 4])
# dropout函数用到的参数
p_keep_input = tf.placeholder('float')
p_keep_hidden = tf.placeholder('float')

# 初始化权重参数
# w1 = tf.Variable(tf.random_normal((2600, 2600)))
# b1 = tf.Variable(tf.random_normal((1, 2600)))
# w2 = tf.Variable(tf.random_normal((2600, 512)))
# b2 = tf.Variable(tf.random_normal((1, 512)))
# w3 = tf.Variable(tf.random_normal((512, 128)))
# b3 = tf.Variable(tf.random_normal((1, 128)))
# w4 = tf.Variable(tf.random_normal((128, 4)))
# b4 = tf.Variable(tf.random_normal((1, 4)))

w1 = tf.Variable(tf.random_normal((100, 1024)))
b1 = tf.Variable(tf.random_normal((1, 1024)))
w2 = tf.Variable(tf.random_normal((1024, 512)))
b2 = tf.Variable(tf.random_normal((1, 512)))
w3 = tf.Variable(tf.random_normal((512, 128)))
b3 = tf.Variable(tf.random_normal((1, 128)))
w4 = tf.Variable(tf.random_normal((128, 4)))
b4 = tf.Variable(tf.random_normal((1, 4)))

# w1 = tf.Variable(tf.random_normal((2600, 512)))
# b1 = tf.Variable(tf.random_normal((1, 512)))
# w2 = tf.Variable(tf.random_normal((512, 256)))
# b2 = tf.Variable(tf.random_normal((1, 256)))
# w3 = tf.Variable(tf.random_normal((256, 128)))
# b3 = tf.Variable(tf.random_normal((1, 128)))
# w4 = tf.Variable(tf.random_normal((128, 4)))
# b4 = tf.Variable(tf.random_normal((1, 4)))

# w1 = tf.Variable(tf.ra

# 计数器变量
global_step = tf.Variable(0, name='global_step', trainable=False)

# 模型计算
# 第一层
X = tf.nn.dropout(X, p_keep_input)
sum1 = tf.matmul(X, w1) + b1
h1 = tf.nn.relu(sum1)
y1 = tf.nn.dropout(h1, p_keep_hidden)
# 第二层
sum2 = tf.matmul(y1, w2) + b2
h2 = tf.nn.relu(sum2)
y2 = tf.nn.dropout(h2, p_keep_hidden)
# 第三层
sum3 = tf.matmul(y2, w3) + b3
h3 = tf.nn.relu(sum3)
y3 = tf.nn.dropout(h3, p_keep_hidden)
# 第四层
sum4 = tf.matmul(y3, w4) + b4

# 计算模型当前的正确率
# y_true = tf.argmax(Y, 1)
y_predict = tf.argmax(sum4, 1)
# correct_prediction = tf.equal(y_predict, y_true)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tf.initialize_all_variables().run()
    ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(len(Index)):
            trX = np.array([np.loadtxt(Test_DATA_FILE_PATH + '\\' + str(Index[i]) + '.txt', delimiter=',')])
            trX = pca.transform(trX)
            y_pred = sess.run(y_predict, feed_dict={X: trX, p_keep_input: KEEP_INPUT, p_keep_hidden: KEEP_HIDDEN})
            print('预测第 %d 条, 预测结果 %s' % (i, CLASSES[y_pred[0]]))
            Result.append([Index[i], CLASSES[y_pred[0]]])

        df = pd.DataFrame(Result, columns=['key', 'predicted class'])
        df.to_csv('./DataSet/Predict/Result_NN_PCA.csv', index=False)


