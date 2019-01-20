import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('./data', one_hot=True)
learn_rate = 0.02
batch_size = 50

x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])
w1 = tf.Variable(tf.random_normal(shape=[784, 144]))
b1 = tf.Variable(tf.constant(0.1, shape=[144]))
h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
w2 = tf.Variable(tf.random_normal(shape=[144, 10]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(h1, w2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

x_axis = []
y_axis = []
fig = plt.figure(figsize=(8,4))
y_t = np.arange(0, 1.1, 0.1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(500000):
    batch = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    if i % 1000 == 0:
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      test = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      print(test)
      x_axis.append(i)
      y_axis.append(test)
      plt.plot(x_axis, y_axis)
      plt.xlabel('Rounds')
      plt.ylabel('accuracy')
      plt.yticks(y_t)
      plt.pause(0.005)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('accuracy: ',accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
