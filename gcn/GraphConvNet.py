from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()  # For easy reset of notebook state.
seed_value = 10
tf.random.set_seed(seed_value)
from SpectralRule import Spectral
import numpy as np

class GraphConvNet(layers.Layer):
    def __init__(self, A, layer_info_list):
        super(GraphConvNet, self).__init__()
        self.adjacency_mat = A
        self.graph_info = layer_info_list

    def build(self, input_shape):
        shp = self.graph_info[-1][0]
        self.w = self.add_weight(shape=(shp, 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.adjacency_mat.shape[0],),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        # loop through the list and build
        x = inputs
        for cnfg in self.graph_info:
            num_feats_per_node, activation = cnfg
            x = Spectral(self.adjacency_mat, num_feats_per_node, activation)(x)

        # now, sum over the features per node and sigmoid
        x = tf.linalg.matmul(x, self.w)
        x = tf.transpose(x) + self.b
        x = tf.keras.activations.sigmoid(x)

        return tf.keras.backend.squeeze(x, axis=0)
# Assuming data is loaded here, making sure A not trainable
A = np.asmatrix([[0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 0, 1, 0]])

# adding self loops
A_hat = np.eye(A.shape[0])
A_hat += A

x = np.asmatrix([[0, 0],
                 [1, -1],
                 [2, -2],
                 [3, -3]])

A_hat = tf.constant(A_hat, dtype=tf.float32)
x = tf.constant(x, dtype=tf.float32)
msk = tf.constant([False, True, True, True])
#msk = None
info_lst = [(10, 'tanh'), (10, 'tanh')]
g = GraphConvNet(A_hat, info_lst)

res = g(x)
lss = tf.keras.losses.BinaryCrossentropy()
# lss(tf.constant([1., 2.]), tf.constant([1., 3.]), sample_weight=tf.constant([2,2]))
print(res)
wts = tf.constant([[0., 1., 1., 1.]])
print(wts.shape)
print(lss(tf.constant([1., 0., 0., 0.]), res, sample_weight=[0., 1., 1., 1.]))

# a loss boi
bce = tf.keras.losses.BinaryCrossentropy()
