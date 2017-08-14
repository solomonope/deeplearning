import tensorflow as tf;
import numpy as NP;

x1 = tf.placeholder(tf.float32);
x2 = tf.placeholder(tf.float32);

print(x1);

sum_op = tf.add(x1, x2);
product_op = tf.multiply(x1, x2);

with tf.Session() as session:
    sum_result = session.run(sum_op, feed_dict={x1: 3, x2: 0.5});
    print(sum_result)
    mult_result = session.run(product_op, feed_dict={x1: 2.0, x2: 0.5});

with tf.Session() as session:
    sum_result = session.run(sum_op, feed_dict={x1: [2.0, 2.0, 2.0], x2: [0.5, 1.0, 2.0]});
    mult_result = session.run(product_op, feed_dict={x1: 2.0, x2: 0.5});
