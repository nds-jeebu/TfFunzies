from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Agent import SmartAgent
import tensorflow as tf





gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
FRAME_STACK_SIZE = 4
MAX_MEMORY_SIZE = 1000000
EXPLORE_PROB_FINAL = 0.01
NUM_TRAINING_EPISODES = 1000
NUM_EPISODES = 75
NUM_RAND_EPS = 15

model_save_path = 'model/'



learner = SmartAgent(FRAME_STACK_SIZE, MAX_MEMORY_SIZE, model_save_path)
learner.setup_DQN(True, model_save_path)
learner.test_policy()
#learner.gather_experience(100000)
