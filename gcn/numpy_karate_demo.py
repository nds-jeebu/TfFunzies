import numpy as np
from networkx import karate_club_graph, to_numpy_matrix
import networkx as nx
import matplotlib.pyplot as plt
# zach's karate graph demo

def relu(A):
    inds = A < 0
    A[inds] = 0

def gcn_layer(A_hat, D_hat, X, W):
    temp = D_hat**-1 * A_hat * X * W
    relu(temp)
    return temp

zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

# apply the rules
A_hat = A + I
D_hat = np.asarray(np.sum(A_hat, axis=0))[0]
D_hat = np.asmatrix(np.diag(D_hat))

# get some random weights
W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))
print('weight layer 1 shape:', np.shape(W_1))
print('weight layer 2 shape:', np.shape(W_2))
print()
print('input feature shape:', np.shape(I))
h_1 = gcn_layer(A_hat, D_hat, I, W_1)
print('first out feature shape:', np.shape(h_1))
h_2 = gcn_layer(A_hat, D_hat, h_1, W_2)
print('final out feature shape:', np.shape(h_2))

feature_representations = {
    node: np.array(h_2)[node]
    for node in zkc.nodes()}

h_2 = np.asarray(h_2)
# plt.scatter(h_2[:, 0], h_2[:, 1])
# plt.show()


# Find key-values for the graph
# pos = nx.spring_layout(zkc)

# Plot the graph
nx.draw(zkc, cmap = plt.get_cmap('rainbow'), with_labels=True, pos=feature_representations)
plt.show()