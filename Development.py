# Created by [Yuexiong Ding] at 2018/2/25
# 测试验证集的准去率
#

import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# CKPT_PATH = './Ckpt_Dir/test5'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_2600_512_128_4'
CKPT_PATH = './Ckpt_Dir/GradientDescent_PCA_100_1024_512_128_4'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_PCA_2600_128_4'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_Test_PCA_100_1024_512_128_4'
# CKPT_PATH = './Ckpt_Dir/adam1'
# DATA_PATH = './DataSet/Normalized/TestData2.txt'
DATA_PATH = './DataSet/Normalized/DevData.txt'
# DATA_PATH = './DataSet/Normalized/TrainingData].txt'
# LABEL_PATH = './DataSet/Normalized/TestLabels2.txt'
LABEL_PATH = './DataSet/Normalized/DevLabels.txt'
# LABEL_PATH = './DataSet/Normalized/TrainingLabels2.txt'

KEEP_INPUT = 1
KEEP_HIDDEN = 1

CLASSES = ['galaxy', 'qso', 'star', 'unknown']

# 加载数据
fix_X = np.loadtxt('./DataSet/Normalized/TrainingData.txt', delimiter=',')
trX = np.loadtxt(DATA_PATH, delimiter=',')
pca = PCA(n_components=100)
pca.fit(fix_X)
DataPCA = pca.fit_transform(trX)
trX = DataPCA
trY = np.loadtxt(LABEL_PATH, delimiter=',')
print('数据加载完毕！')

# X = tf.placeholder('float', [None, 2600])
X = tf.placeholder('float', [None, 100])
# X = tf.placeholder('float', [None, 5])
Y = tf.placeholder('float', [None, 4])
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

# w1 = tf.Variable(tf.random_normal((100, 128)))
# b1 = tf.Variable(tf.random_normal((1, 128)))
# w2 = tf.Variable(tf.random_normal((128, 4)))
# b2 = tf.Variable(tf.random_normal((1, 4)))

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
y_true = tf.argmax(Y, 1)
y_predict = tf.argmax(sum4, 1)
correct_prediction = tf.equal(y_predict, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tf.initialize_all_variables().run()
    ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        train_accuracy, y_true, y_predict = sess.run([accuracy, y_true, y_predict],
                                                     feed_dict={X: trX, Y: trY, p_keep_input: KEEP_INPUT,
                                                                p_keep_hidden: KEEP_HIDDEN})
        print(y_true)
        print(y_predict)
        print("development accuracy %g" % train_accuracy)
        print("micro recall accuracy %g" % metrics.recall_score(y_true, y_predict, average='micro'))
        print("macro recall accuracy %g" % metrics.recall_score(y_true, y_predict, average='macro'))
        print("confusion_matrix:")
        print(confusion_matrix(y_true, y_predict))

        print("classification_report:")
        print(classification_report(y_true, y_predict, target_names=CLASSES))
        print("f1 accuracy %g" % metrics.f1_score(y_true, y_predict, average='weighted'))
