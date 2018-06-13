import random
import numpy as np

class QuadcopterAgent:

    def __init__(self,task,buffer_size=32,gamma=0.95,min_epsilon=0.01,epsilon_decay=0.995,epsilon=1.0):
        self.task = task
        self.buffer = []
        self.buffer_size=buffer_size
        self.gamma=gamma
        self.min_epsilon=min_epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon=epsilon

    def build_model(self):
        pass

    def act(self):
        return [i/12 for i in range(1,13)]

    def remember(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self):
        assert self.buffer_size>=len(self.buffer)
        return random.sample(self.buffer,self.buffer_size)

    def replay(self):
        batch = self.sample()
        for state,action,reward,next_state,done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

