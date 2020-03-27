import numpy as np
import cv2
from collections import deque


class Util:

    @staticmethod
    def preprocess_frame(frame):
        # Convert to grayscale and downscale
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, (110, 84))
        return img/255.

    @staticmethod
    def new_frame_stack(stack_size):
        return deque(maxlen=stack_size)

    @staticmethod
    def stack_frames(frame_stack, stack_size, new_frame, is_new_episode):
        frame = Util.preprocess_frame(new_frame)

        if is_new_episode:
            # Clear all other stacked frames
            frame_stack = deque(maxlen=stack_size)

            # Now, paste the new frame in 4 times
            for _ in range(stack_size):
                frame_stack.append(frame)
        else:
            # Otherwise, stack the new frame
            frame_stack.append(frame)

        # Return the MxNxstack_size array
        return np.stack(frame_stack, axis=2), frame_stack
