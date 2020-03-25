from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import datetime


tf.keras.backend.clear_session()
# a layer encapsulates the weights and the computation

# A: adjacency mat with identity tagged already
# out_n: number of output features per node
# activation: choice of activation function
class Spectral(layers.Layer):
    def __init__(self, A, out_n, activation):
        super(Spectral, self).__init__()
        self.out_n = out_n
        self.activation = activation
        self.adjacency_mat = A
        deg_mat = tf.reduce_sum(A, axis=0)
        deg_mat_inv = tf.math.pow(deg_mat, -.5)
        deg_mat_inv = tf.linalg.tensor_diag(deg_mat_inv)
        self.deg_mat_inv = deg_mat_inv

    def build(self, input_shape):
        # here's this in case we want to explicitly assign
        #init = tf.keras.initializers.Constant(value=[[1,-1],[-1,1]])
        self.w = self.add_weight(shape=(input_shape[-1], self.out_n),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs, **kwargs):
        aggregate = tf.linalg.matmul(self.deg_mat_inv, self.adjacency_mat)
        aggregate = tf.linalg.matmul(aggregate, self.deg_mat_inv)
        aggregate = tf.linalg.matmul(aggregate, inputs)
        aggregate = tf.linalg.matmul(aggregate, self.w)

        if self.activation == 'leaky_relu':
            act = tf.keras.layers.LeakyReLU(.3)
        elif self.activation == 'tanh':
            act = tf.keras.layers.Activation(tf.nn.tanh)
        else:
            act = tf.keras.layers.ReLU()
        return act(aggregate)


# A = np.asmatrix([[0, 1, 0, 0],
#                 [0, 0, 1, 1],
#                 [0, 1, 0, 0],
#                 [1, 0, 1, 0]])
#
# # adding self loops
# A_hat = np.eye(A.shape[0])
# A_hat += A
#
# x = np.asmatrix([[0, 0],
#                  [1, -1],
#                  [2, -2],
#                  [3, -3]])
#
# A_hat = tf.constant(A_hat, dtype=tf.float32)
# x = tf.constant(x, dtype=tf.float32)
#
# spec = Spectral(A_hat, 2, '')(x)
#
# a = tf.ones(shape=(spec.shape[-1],1))
# res = tf.linalg.matmul(spec, a)
# print(spec)
#
# print(tf.transpose(res) + tf.ones(shape=(4,)))
# def create_model():
#     return tf.keras.models.Sequential([
#         Spectral(A_hat, 2, '')
#     ])
#
# model = create_model()
# log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# # apply the layer
