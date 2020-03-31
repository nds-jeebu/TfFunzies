import tensorflow as tf
import numpy as np

#######################

x = tf.constant(3.)

with tf.GradientTape() as tape:
    # There's only need to watch x because it is a constant
    tape.watch(x)
    y = x**3

print(tape.gradient(y, x).numpy())

# In the case where x is a variable, this is automatic
x = tf.Variable(3., trainable=True)
with tf.GradientTape() as tape:
    y = x**3

print(tape.gradient(y, x).numpy())

# If we don't want the gradient tape to watch certain vars:
with tf.GradientTape(watch_accessed_variables=False) as tape:
    y = x**3
print(tape.gradient(y, x))

# To get higher order derivatives, nest the calls
x = tf.Variable(4., trainable=True)

with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        y = x**3
        order1 = tape2.gradient(y, x)
    order2 = tape1.gradient(order1, x)

print(order1.numpy())
print(order2.numpy())


# By default, the gradient tape throws away all mem after called
a = tf.Variable(6., trainable=True)
b = tf.Variable(2., trainable=True)

with tf.GradientTape(persistent=True) as tape:
    y1 = a**2
    y2 = b**3

print(tape.gradient(y1, a).numpy())
print(tape.gradient(y2, b).numpy())