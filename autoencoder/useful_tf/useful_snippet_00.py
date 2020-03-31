import tensorflow as tf
import numpy as np

tf.random.set_seed(0)
######### demo usage of tf.reduce_sum ############
# I really want to understand how to use batches

# Assume the leading dimension is the batch dimension
dummy = np.array([np.eye(3), np.eye(3), np.eye(3)])
#a = tf.random.normal(shape=(4,3,2))
a = tf.constant(dummy)
b = tf.reduce_sum(a, axis=[1,2])
print(a)
print('reducing the sum along axis 1:', b)

c = tf.reduce_mean(a)
# mean takes the entire value
print('reduce mean of a:', c)