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
        self.lay_lst = []
        for cnfg in layer_info_list:
            num_feats_per_node, activation = cnfg
            self.lay_lst.append(Spectral(self.adjacency_mat, num_feats_per_node, activation))

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
        for i in range(len(self.lay_lst)):
            x = self.lay_lst[i](x)
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

x = np.asmatrix([[0., 0.],
                 [1., -1.],
                 [2., -2.],
                 [3., -3.]])

A_hat = tf.constant(A_hat, dtype=tf.float32)
feat = tf.constant(x, dtype=tf.float32)
# msk = tf.constant([False, True, True, True])
# #msk = None
info_lst = [(10, 'tanh'), (10, 'tanh')]
#g = GraphConvNet(A_hat, info_lst)
#
#res = g(feat)
# lss = tf.keras.losses.BinaryCrossentropy()
# # lss(tf.constant([1., 2.]), tf.constant([1., 3.]), sample_weight=tf.constant([2,2]))
# print(res)
#print(tf.keras.backend.gather(res, [0,3]))

#print(lss(, [0., 1.]))
# wts = tf.constant([[0., 1., 1., 1.]])
# print(wts.shape)
# print(lss(tf.constant([1., 0., 0., 0.]), res, sample_weight=[0., 1., 1., 1.]))
class MyModel(tf.keras.Model):
    def __init__(self, A_init, opt_list):
        super(MyModel, self).__init__()
        self.gcn = GraphConvNet(A_init, opt_list)

    def call(self, inputs, training=None, mask=None):
        k = self.gcn(inputs)
        k = tf.keras.backend.gather(k, [0,3])
        return k

model = MyModel(A_hat, info_lst)
# inputs = tf.keras.Input(shape=(4,2))
# k = GraphConvNet(A_hat, info_lst)(inputs)
# outputs = tf.keras.backend.gather(k, [0,3])
#
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
model = MyModel(A_hat, info_lst)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy()
)

y_t = tf.constant([0.,1.])
for i in range(2000):
    model.train_on_batch(x=feat, y=y_t)

print(model.predict_on_batch(x))
# inputs = tf.keras.Input(shape=(4,2), name='feats')
# x = GraphConvNet(A_hat, info_lst)(inputs)
# outputs = tf.keras.backend.gather(x, [0,3])
#
# model = tf.keras.Model()
# model.
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#               metrics=['accuracy'])
#
# test_loss, test_acc = model.evaluate(np.array([feat]),  np.array([[0.,0.]]), verbose=2)