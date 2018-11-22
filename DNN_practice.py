#!/usr/bin/env python
# encoding: utf-8
'''
@author: Great
@software: garner
@file: DNN_practice.py
@time: 2018/11/21 17:17
'''
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.15, random_state= 0 )

X_train = scale(X_train)
X_test = scale(X_test)
y_train = scale(y_train.reshape((-1,1)))
y_test = scale(y_test.reshape((-1,1)))

xs = tf.placeholder(tf.float32, shape=[None,X_train.shape[1]], name="inputs")
ys = tf.placeholder(tf.float32, shape=[None,1], name = "y")
prob_s = tf.placeholder(tf.float32)

hidden1 = 20
hidden2 = 10
learning_rate = 0.03
epoch = 2000

def add_layer(inputs,inputs_size, outputs_size,activation_func = None ):

    with tf.variable_scope("Weights"):
        Weight = tf.Variable(tf.random_normal([inputs_size, outputs_size]), name="weights")
    with tf.variable_scope("Biases"):
        Biases = tf.Variable(tf.zeros([1,outputs_size]), name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weight) + Biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob= prob_s)

    if activation_func is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_func"):
            return activation_func(Wx_plus_b)

with tf.name_scope("layer1"):
    layer1 = add_layer(xs,X_train.shape[1],hidden1,activation_func=tf.nn.relu)
with tf.name_scope("layer2"):
    layer2 = add_layer(layer1, hidden1, hidden2, activation_func = tf.nn.relu)
with tf.name_scope("y_pred"):
    output = add_layer(layer2, hidden2, 1, activation_func= None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - output), reduction_indices=[1]))
    tf.summary.scalar("loss",tensor = loss)
with tf.name_scope("train"):
    train_opt = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss)

def train(X, y, n ):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=15)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="boston_log",graph=sess.graph)
        for i in range(n):

            sess.run(train_opt, feed_dict= {xs:X,ys:y, prob_s: 1})
            if i%100 == 0:
                _loss = sess.run(loss, feed_dict= {xs:X,ys:y, prob_s: 1})

                print(i, _loss)
                rs = sess.run(merged, feed_dict={xs:X,ys:y, prob_s:1})
                writer.add_summary(summary=rs, global_step=i)
        saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=i)



train(X_train,y_train,epoch)

