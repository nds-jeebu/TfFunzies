import tensorflow as tf
import numpy as np
import random

def loss(real_y, pred_y):
    return tf.reduce_mean(tf.abs(real_y - pred_y))

x_train = np.array([[1,2,3],[4,5,6],[7,8,9]])
y_train = np.array([[4, 7, 10], [13, 16, 19], [22, 25, 28]])

a = tf.Variable(random.random(), trainable=True)
b = tf.Variable(random.random(), trainable=True)

# training

def train_step(real_x, real_y):
    with tf.GradientTape() as tape:
        pred_y = a * real_x + b

        lss = loss(real_y, pred_y)
        print('loss:', lss)

    a_grad, b_grad = tape.gradient(lss, (a,b))
    a.assign_sub(a_grad * 0.001)
    b.assign_sub(b_grad * 0.001)

    print('a_grad:', a_grad.numpy())
    print('b_grad:', b_grad.numpy())

for i in range(100):
    train_step(x_train, y_train)
    print(a.numpy())
    print(b.numpy())