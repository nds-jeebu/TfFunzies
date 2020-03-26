from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.keras import layers

class DQNModel:
    def __init__(self, num_possible_actions):
        self.num_actions = num_possible_actions

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(16, (8, 8), activation='relu', input_shape=(110, 84, 4)))
        model.add(layers.Conv2D(32, (4, 4), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.num_actions))