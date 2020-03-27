from collections import deque
import numpy as np


class Memory:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]

# tester = Memory(5)
# tester.add([1])
# tester.add([14])
# tester.add([3])
# tester.add([2])
# tester.add([1])
#
# print(tester.sample(3))
