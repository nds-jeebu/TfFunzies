from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


from datetime import datetime
from packaging import version
class DQNModel:

    def __init__(self, num_possible_actions, stack_size):
        self.num_actions = num_possible_actions
        self.stack_size = stack_size
        self.target_model = None
        self.prediction_model = None

    def build_prediction_model(self):
        self.prediction_model = self.build_based_model()
        # model.Inputs
        self.prediction_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.Huber())

    def build_target_model(self):
        self.target_model = self.build_based_model()

    def update_target_model(self):
        wts_pred = self.prediction_model.get_weights()
        wts_targ = self.target_model.get_weights()

        for i in range(len(wts_pred)):
            wts_targ[i] = wts_targ[i] * .95 + wts_pred[i] * .05

        self.target_model.set_weights(wts_targ)

    def build_based_model(self):
        model = tf.keras.models.Sequential()
        #model.add(layers.Lambda(lambda x: x/255.))
        model.add(layers.Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(110, 84, self.stack_size)))
        model.add(layers.Conv2D(32, (4, 4), strides=2, activation='relu'))
        model.add(layers.Conv2D(32, (4, 4), strides=2, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.num_actions))

        return model

    def restore_model(self, model_path):
        self.prediction_model = tf.keras.models.load_model(model_path)
        self.target_model = self.build_based_model()
        self.target_model.set_weights(self.prediction_model.get_weights())

    def process_mem_block(self, mem_block, discount):
        train_input_batch = []
        train_targ_batch = []
        for state, action, reward, next_state, is_term in mem_block:
            # The input will be the frame_stack
            train_input_batch.append(state)
            # Determine if state was terminal
            target_q_value = 0
            state = np.array([state])
            if not is_term:
                next_state = np.array([next_state])

                # For double DQN, we need to update this section
                #intermed_q_vals = np.array(self.target_model.predict_on_batch(next_state))
                #future_disc_reward = discount * np.max(intermed_q_vals[0])
                target_net_action = np.array(self.target_model.predict_on_batch(state))[0]
                target_net_action = np.argmax(target_net_action)
                target_q_value = np.array(self.prediction_model.predict_on_batch(next_state))[0]
                target_q_value = target_q_value[target_net_action]

            expected_q = reward + discount * target_q_value

            # Now, we want to set it equal to the associated target q value
            #state = np.array([state])
            target = np.array(self.prediction_model.predict_on_batch(state))[0]
            target[action] = expected_q
            train_targ_batch.append(target)

        return np.array(train_input_batch), np.array(train_targ_batch)

    def train(self, mem_block, discount, num_epochs):
        train_input_batch, train_targ_batch = self.process_mem_block(mem_block, discount)

        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        history = self.prediction_model.fit(train_input_batch, train_targ_batch, epochs=num_epochs, verbose=0)
                                 #callbacks=[tensorboard_callback])

    def get_next_action(self, curr_state, explore_prob):

        if np.random.rand() < explore_prob:
            action = np.random.randint(self.num_actions)
        else:
            curr_state = np.array([curr_state])
            q_vals = np.array(self.prediction_model.predict_on_batch(curr_state))[0]

            #print(q_vals)
            # plt.imshow(curr_state[0][:,:,3])
            # plt.show()
            #print('q_vals:', q_vals)
            action = np.argmax(q_vals)
        return action

    def save_model_params(self, save_path):
        self.prediction_model.save(save_path)
