# Import the gym module
import gym
import matplotlib.pyplot as plt
import cv2
# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False
while not is_done:
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    plt.show()


# 1. figure out the action replay thing
# 2. figure out the RL part
# 3. build the CNN
# 4. create the training loop