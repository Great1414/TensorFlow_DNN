#!/usr/bin/env python
# encoding: utf-8
'''
@author: Great
@time: 2018/11/19 14:13
'''
import tensorflow as tf
import math
#1设置网络参数
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
#2设置占位符
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

#3神经网络的层次
#当前网络层，X输入层，n_neurons输出层，activation激活函数
def neuron_layer(X, n_neurons, name, activation=None):  # 根据介绍，按照此方法计算标准差，可以加快收敛
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / math.sqrt(n_inputs)
        #产生一个具有截断的均值，标准差可以设定的高斯分布数据，如截断（-3std, +3std）
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        #设置权重和偏置，权重为随机选取的正太分布，偏置设为0
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        #计算待输出结果
        z = tf.matmul(X, W) + b
        #是否要激活函数计算
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

#4设置层次结构
#使用with句式方便管理、可视化，代码整洁
with tf.name_scope("dnn"):
    #隐藏层1，激活函数为relu
    hidden1 = neuron_layer(X, n_hidden1, "n_hidden1", activation="relu")
    #隐藏层2，激活函数为relu, 注意输入变为了hidden1
    hidden2 = neuron_layer(hidden1, n_hidden2, "n_hidden2", activation="relu")
    #输出层，激活函数暂时不设置
    logits = neuron_layer(hidden2, n_outputs, "n_outputs")

#5损失计算
#利用交叉熵计算loss
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

#6梯度下降优化
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

#7准确度计算
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#8导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

#9训练
#设置迭代和小批量大小
n_epoch = 400
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        for batch in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        if epoch % 10 == 0:
            loss_ = sess.run(loss, feed_dict={X: X_batch, y: y_batch})
            print(epoch, "loss: {:.4f}".format(loss_), "train accuracy: {:.3f}".format(acc_train),
                  "test accuracy: {:.3f}".format(acc_test))
    #保存模型参数，方便后续直接调用
    save_path = saver.save(sess, "./final_model.ckpt")