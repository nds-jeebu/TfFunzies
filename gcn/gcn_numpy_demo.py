import numpy as np
from scipy.linalg import fractional_matrix_power
import tensorflow as tf
# Here's the first demo in implementing this gcn

# adjacency matrix
A = np.asmatrix([[0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 0, 1, 0]])

# adding self loops
A_hat = np.eye(A.shape[0])
A_hat += A

# features per node
x = np.asmatrix([[0, 0],
                 [1, -1],
                 [2, -2],
                 [3, -3]])

# propagation rule: A * x <-- add all nghbr feats and self



# degree matrix
D = np.array(np.sum(A_hat, axis=0))

D = np.asmatrix(np.diag(D))


# weight matrix
w = np.asmatrix([[1, -1],
                 [-1, 1]])
# normalized prop rule

# d_tf = tf.constant(D**-1, dtype=tf.float32)
# a_tf = tf.constant(A_hat, dtype=tf.float32)
# x_tf = tf.constant(x, dtype=tf.float32)
# w_tf = tf.constant(w, dtype=tf.float32)
#
# res_tf = tf.matmul(d_tf, a_tf)
# res_tf = tf.matmules_tf, x_tf)
# res_tf = tf.matmul(res_tf, w_tf)
#
# print(res_tf)

temp = np.matmul(fractional_matrix_power(D, -.5), A_hat)
temp = np.matmul(temp, fractional_matrix_power(D, -.5))
temp = np.matmul(temp, x)
temp = np.matmul(temp, w)
print(temp)

# at this point we can apply some activation function