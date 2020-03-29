from Memory import Memory
import gym
from util import Util
import numpy as np
from DQNModel import DQNModel
import matplotlib.pyplot as plt

class SmartAgent:

    def __init__(self, stack_size, max_memory_size, model_save_path):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.memory = Memory(max_memory_size)
        self.frame_stack_size = stack_size
        self.explore_prob = .01
        self.explore_prob_final = 0.01
        self.explore_decay = .995
        self.DQN = DQNModel(self.env.action_space.n, stack_size)
        self.num_exps = 0
        self.model_save_path = model_save_path
        self.discount = .99

    def setup_DQN(self, load_existing=None, model_path=None):
        if load_existing:
            self.DQN.restore_model(model_path)
        else:
            self.DQN.build_target_model()
            self.DQN.build_prediction_model()
            self.DQN.save_model_params(self.model_save_path)

    def train(self, batch_size, num_epochs, update_target):
        mem_block = self.memory.sample(batch_size)
        self.DQN.train(mem_block, self.discount, num_epochs)
        if self.num_exps % 1000 == 0:
            self.DQN.save_model_params(self.model_save_path)
        if update_target:
            self.DQN.update_target_model()
        if self.explore_prob > self.explore_prob_final:
            self.explore_prob -= (1 - .01)/1e5

    def test_policy(self):
        is_new_episode = True
        frame_stack = Util.new_frame_stack(self.frame_stack_size)
        first_frame = self.env.reset()
        # self.env.render()
        state, frame_stack = Util.stack_frames(frame_stack, self.frame_stack_size, first_frame, is_new_episode)
        episode_is_done = False
        while not episode_is_done:
            print(self.num_exps)
            is_new_episode = False

            # Take an action
            num_acts = 0
            action = self.DQN.get_next_action(state, self.explore_prob)

            frame, reward, episode_is_done, _ = self.env.step(action)
            self.env.render()

            next_state, frame_stack = Util.stack_frames(frame_stack, self.frame_stack_size, frame, is_new_episode)

            state = next_state

    def gather_experience(self, num_training):
        for i in range(num_training):
            pts = 0
            episode_is_done = False
            is_new_episode = True
            # Start the environment and put the first frame into a stack
            frame_stack = Util.new_frame_stack(self.frame_stack_size)
            first_frame = self.env.reset()
            #self.env.render()
            state, frame_stack = Util.stack_frames(frame_stack, self.frame_stack_size, first_frame, is_new_episode)
            # for i in range(4):
            #3plt.imshow(state[:,:,0])
            #plt.show()
            #print(state.shape)
            num_consec_actions = 4
            num_acts = 0
            while not episode_is_done:
                print(self.num_exps)
                is_new_episode = False

                # Take an action

                if num_acts % num_consec_actions == 0:
                    num_acts = 0
                    action = self.DQN.get_next_action(state, self.explore_prob)

                frame, reward, episode_is_done, _ = self.env.step(action)
                pts += reward
                num_acts += 1
                #print(reward)
                #plt.imshow(frame)
                #plt.show()
                #self.env.render()
                next_state, frame_stack = Util.stack_frames(frame_stack, self.frame_stack_size, frame, is_new_episode)
                # for i in range(4):
                #     plt.imshow(next_state[:, :, i])
                #     plt.show()
                experience = (state, action, np.sign(reward), next_state, episode_is_done)
                state = next_state
                self.memory.add(experience)
                self.num_exps += 1

                if self.num_exps >= 50000:
                    if (self.num_exps - 200) % 4 == 0:
                        if (self.num_exps - 200) % 8 == 0:
                            self.train(32, 1, True)
                        else:
                            self.train(32, 1, False)
                #print('Explore Prob:', self.explore_prob)


            print('Total points:', pts)










