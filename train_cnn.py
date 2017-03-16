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
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    learning_rate = 0.001
    dropout = 0.75
    batch_size = 100
    x_ = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(dtype=tf.int64, shape=[None])
    prob = tf.placeholder(dtype=tf.float32)

    with tf.variable_scope("CNN"):

        W1 = tf.get_variable(name="weight_1", shape=[
            5, 5, 1, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(name="weight_2", shape=[
            5, 5, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable(name="weight_3", shape=[
            7 * 7 * 64, 1024], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        W4 = tf.get_variable(name="weight_4", shape=[
            1024, 10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable(name="biases_1", shape=[32],
                             initializer=tf.constant_initializer(0.01))
        b2 = tf.get_variable(name="biases_2", shape=[64],
                             initializer=tf.constant_initializer(0.01))
        b3 = tf.get_variable(name="biases_3", shape=[1024],
                             initializer=tf.constant_initializer(0.01))
        b4 = tf.get_variable(name="biases_4", shape=[10],
                             initializer=tf.constant_initializer(0.01))

        strides = 1
        k = 2
        conv1 = tf.nn.conv2d(x_, W1, strides=[1, strides, strides, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        conv2 = tf.nn.conv2d(conv1, W2, strides=[1, strides, strides, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        fc1 = tf.reshape(conv2, shape=[-1, W3.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, W3), b3)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, prob)

        out = tf.add(tf.matmul(fc1, W4), b4)

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=out, labels=y_, name="cross_entropy"))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        correct_pred = tf.equal(tf.argmax(out, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(1, 11):
            batches = int(np.ceil(len(trainX) / batch_size))

            for j in range(batches):
                start = j * batch_size
                end = (j + 1) * batch_size if (j + 1) * batch_size < len(trainX) else len(trainX)
                sess.run(train_step, feed_dict={x_: trainX[
                         start:end], y_: trainY[start:end], prob: dropout})

            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x_: trainX[
                                 start:end], y_: trainY[start:end], prob: 1.})
            print(i, "Loss =", loss, "Accuracy =", acc)

            save_path = saver.save(sess, "./weights/cnn.ckpt")

        save_path = saver.save(sess, "./weights/cnn.ckpt")


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
    prob = tf.placeholder(dtype=tf.float32)

    with tf.variable_scope("CNN") as scope:

        try:
            W1 = tf.get_variable(name="weight_1", shape=[
                                 5, 5, 1, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            scope.reuse_variables()
            W1 = tf.get_variable(name="weight_1", shape=[
                                 5, 5, 1, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        W2 = tf.get_variable(name="weight_2", shape=[
            5, 5, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable(name="weight_3", shape=[
            7 * 7 * 64, 1024], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        W4 = tf.get_variable(name="weight_4", shape=[
            1024, 10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable(name="biases_1", shape=[32],
                             initializer=tf.constant_initializer(0.01))
        b2 = tf.get_variable(name="biases_2", shape=[64],
                             initializer=tf.constant_initializer(0.01))
        b3 = tf.get_variable(name="biases_3", shape=[1024],
                             initializer=tf.constant_initializer(0.01))
        b4 = tf.get_variable(name="biases_4", shape=[10],
                             initializer=tf.constant_initializer(0.01))

        strides = 1
        k = 2
        conv1 = tf.nn.conv2d(x_, W1, strides=[1, strides, strides, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        conv2 = tf.nn.conv2d(conv1, W2, strides=[1, strides, strides, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        fc1 = tf.reshape(conv2, shape=[-1, W3.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, W3), b3)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, prob)

        out = tf.add(tf.matmul(fc1, W4), b4)

        correct_prediction = tf.argmax(out, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./weights/cnn.ckpt")
        length = len(testX)
        res1 = sess.run(correct_prediction, feed_dict={x_: testX[:int(length / 4)], prob: 1.})
        res2 = sess.run(correct_prediction,
                        feed_dict={x_: testX[int(length / 4):int(2 * length / 4)], prob: 1.})
        res3 = sess.run(correct_prediction,
                        feed_dict={x_: testX[int(2 * length / 4):int(3 * length / 4)], prob: 1.})
        res4 = sess.run(correct_prediction, feed_dict={x_: testX[int(3 * length / 4):], prob: 1.})

        res = np.concatenate((res1, res2, res3, res4))

    return res
