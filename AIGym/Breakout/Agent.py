from Memory import Memory
import gym
import Util
import numpy as np

STACK_SIZE = 4
MAX_MEMORY_SIZE = 100000


class SmartAgent:

    def __init__(self, stack_size, max_memory_size):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.memory = Memory(max_memory_size)
        self.frame_stack_size = stack_size

    # Perform frame stacking (4) and also frame skipping (4)
    def init_fill_memory(self, pretraining_len):

        for i in range(pretraining_len-1):

            if i == 0:
                is_new_frame = True
                # Start the environment and put the first frame into a stack
                frame_stack = Util.new_frame_stack(self.frame_stack_size)
                first_frame = self.env.reset()
                state, frame_stack = Util.stack_frames(frame_stack, first_frame, is_new_frame)

            # Take a random action
            act = self.env.action_space.sample()
            frame, reward, is_done, _ = self.env.step(act)

            # Finished if agent dies three times
            if is_done:
                next_state = np.zeros(state.shape, dtype=np.float32)
                self.memory.add((state, act, reward, next_state, is_done))

                # Create a new episode
                is_new_frame = True
                first_frame = self.env.reset()
                state, frame_stack = Util.stack_frames(frame_stack, first_frame, is_new_frame)
            else:
                # Collect only every 4th frame
                if (i+1) % 4 == 0:
                    is_new_frame = False
                    next_state, frame_stack = Util.stack_frames(frame_stack, frame, is_new_frame)
                    self.memory.add((state, act, reward, next_state, is_done))
                    state = next_state











