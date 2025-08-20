import collections
import random as rand
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    
    def add(self, state, action, reward, next_state, done, skill):
        self.buffer.append((state, action, reward, next_state, done, skill))
    
    def sample(self, batch_size):
        transitions = rand.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, skill = zip(*transitions)
        return(
            np.array(state),
            np.array(action),
            np.array(reward).reshape(-1,1),
            np.array(next_state),
            np.array(done).reshape(-1,1),
            np.array(skill)
        )
    def __len__(self):
        return len(self.buffer)