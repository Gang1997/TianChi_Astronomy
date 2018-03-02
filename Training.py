# Created by [Yuexiong Ding] at 2018/2/25
# TensorFlow 训练模型
#

import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CKPT_PATH = './Ckpt_Dir/test3'
# CKPT_PATH = './Ckpt_Dir/test5'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_2600_512_128_4'
CKPT_PATH = './Ckpt_Dir/GradientDescent_PCA_100_1024_512_128_4'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_PCA_2600_128_4'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_PCA_100_1024_512_128_4'
# CKPT_PATH = './Ckpt_Dir/GradientDescent_Test_PCA_100_1024_512_128_4'
DATA_PATH = './DataSet/Normalized/TrainingData.txt'
# DATA_PATH = './DataSet/Normalized/DevData.txt'
# DATA_PATH = './DataSet/Normalized/TestData.txt'
# DATA_PATH = './DataSet/Normalized/TestData2.txt'
LABEL_PATH = './DataSet/Normalized/TrainingLabels.txt'
# LABEL_PATH = './DataSet/Normalized/DevLabels.txt'
# LABEL_PATH = './DataSet/Normalized/TestLabels.txt'
# LABEL_PATH = './DataSet/Normalized/TestLabels2.txt'
# dropout率
KEEP_INPUT = 1
KEEP_HIDDEN = 1
# 训练次数
ITERATOR = 100000

# batch_size
# BATCH_SIZE = 64
# BATCH_SIZE = 256
# BATCH_SIZE = 50
# BATCH_SIZE = 100
# BATCH_SIZE = 200
# BATCH_SIZE = 1024
# BATCH_SIZE = 500
# BATCH_SIZE = 2000
BATCH_SIZE = 1196
# BATCH_SIZE = 4784
# BATCH_SIZE = 11965


# 加载数据
trX = np.loadtxt(DATA_PATH, delimiter=',')
pca = PCA(n_components=100)
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

# 代价敏感因子
w_ls = tf.Variable([[4], [10], [1], [0.1]], name="w_ls", trainable=False)
# w_ls = tf.placeholder('float', [4, 1])

# 初始化权重参数
# w1 = tf.Variable(tf.random_normal((2600, 1024)))
# b1 = tf.Variable(tf.random_normal((1, 1024)))
# w2 = tf.Variable(tf.random_normal((1024, 512)))
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

# w1 = tf.Variable(tf.random_normal((2600, 2600)))
# b1 = tf.Variable(tf.random_normal((1, 2600)))
# w2 = tf.Variable(tf.random_normal((2600, 512)))
# b2 = tf.Variable(tf.random_normal((1, 512)))
# w3 = tf.Variable(tf.random_normal((512, 128)))
# b3 = tf.Variable(tf.random_normal((1, 128)))
# w4 = tf.Variable(tf.random_normal((128, 4)))
# b4 = tf.Variable(tf.random_normal((1, 4)))

# w1 = tf.Variable(tf.random_normal((2600, 512)))
# b1 = tf.Variable(tf.random_normal((1, 512)))
# w2 = tf.Variable(tf.random_normal((512, 256)))
# b2 = tf.Variable(tf.random_normal((1, 256)))
# w3 = tf.Variable(tf.random_normal((256, 128)))
# b3 = tf.Variable(tf.random_normal((1, 128)))
# w4 = tf.Variable(tf.random_normal((128, 4)))
# b4 = tf.Variable(tf.random_normal((1, 4)))

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

# 计算损失(代价损失)
w_temp = tf.matmul(Y, w_ls)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sum4, labels=Y))
cost = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=sum4, labels=Y), w_temp))
train_op = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
# train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)

# 计算模型当前的正确率
correct_prediction = tf.equal(tf.argmax(sum4, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建存储模型路径
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tf.initialize_all_variables().run()
    ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    start = global_step.eval()
    print("start from:", str(start))

    for i in range(start, ITERATOR):
        for (start, end) in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX) + 1, BATCH_SIZE)):
            batch_X = trX[start: end]
            batch_Y = trY[start: end]
            sess.run(train_op,
                     feed_dict={X: batch_X, Y: batch_Y, p_keep_input: KEEP_INPUT, p_keep_hidden: KEEP_HIDDEN})

            train_accuracy, Cost = sess.run([accuracy, cost],
                                            feed_dict={X: batch_X, Y: batch_Y, p_keep_input: KEEP_INPUT,
                                                       p_keep_hidden: KEEP_HIDDEN})
            print("step %d, cost: %g, training accuracy: %g" % (i, Cost, train_accuracy))

            global_step.assign(i).eval()
            saver.save(sess, CKPT_PATH + "/model.ckpt", global_step=global_step)
