import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plot;

np.random.seed(42);

n_input = 784;

n_dense = 128;

x = tf.placeholder(tf.float32, shape=[None, 784])
b = tf.Variable(tf.zeros([n_dense]));
W = tf.Variable(tf.random_uniform([n_input, n_dense]))

z = tf.add(tf.matmul(x,W), b);
a  = tf.sigmoid(z);

init_op = tf.global_variables_initializer();

session =  tf.Session();
session.run(init_op)
output = session.run(a, feed_dict={x: np.random.random((1, n_input))})


session.close();
print(output.shape)
print(output)