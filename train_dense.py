'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Dibya Jyoti Roy
Roll No.: 13CS10020

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
import numpy as np
import tensorflow as tf
import os


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    # Create the model
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    learning_rate = 0.01
    batch_size = 100
    x_ = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    x = tf.reshape(x_, shape=[-1, 784])
    y_ = tf.placeholder(dtype=tf.int32, shape=[None])

    with tf.variable_scope("DNN"):

        W = tf.get_variable(name="weight_1", shape=[
                            784, 200], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases_1", shape=[200],
                            initializer=tf.constant_initializer(0.01))
        h = tf.nn.relu6(tf.matmul(x, W) + b)

        U = tf.get_variable(name="weight_2", shape=[
                            200, 10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        c = tf.get_variable(name="biases_2", shape=[10], initializer=tf.constant_initializer(0.01))
        o = tf.matmul(h, U) + c

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=o, labels=y_, name="cross_entropy"))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(1, 101):
            tot_loss = 0
            batches = int(np.ceil(len(trainX) / batch_size))
            for j in range(batches):
                start = j * batch_size
                end = (j + 1) * batch_size if (j + 1) * batch_size < len(trainX) else len(trainX)
                _, loss = sess.run([train_step, cross_entropy], feed_dict={
                                   x_: trainX[start:end], y_: trainY[start:end]})
                tot_loss = tot_loss + loss
            print(i, "Loss =", tot_loss / batches)
            if i % 20 == 0:
                save_path = saver.save(sess, "./weights/dnn.ckpt")
        save_path = saver.save(sess, "./weights/dnn.ckpt")


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    x_ = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    x = tf.reshape(x_, shape=[-1, 784])

    with tf.variable_scope("DNN") as scope:

        try:
            W = tf.get_variable(name="weight_1", shape=[
                784, 200], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            scope.reuse_variables()
            W = tf.get_variable(name="weight_1", shape=[
                784, 200], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name="biases_1", shape=[200],
                            initializer=tf.constant_initializer(0.1))
        h = tf.nn.relu6(tf.matmul(x, W) + b)

        U = tf.get_variable(name="weight_2", shape=[
                            200, 10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        c = tf.get_variable(name="biases_2", shape=[10], initializer=tf.constant_initializer(0.1))
        o = tf.matmul(h, U) + c

        correct_prediction = tf.argmax(o, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./weights/dnn.ckpt")
        res = sess.run(correct_prediction, feed_dict={x_: testX})

    return res
