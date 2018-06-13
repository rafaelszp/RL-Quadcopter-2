import sys
import unittest
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from task import Task
import numpy as np
from agents.agent import DDPGAgent

class DDPGTestCase(unittest.TestCase):

    def setUp(self):
        self.num_episodes = 50
        self.target_pos = np.array([0., 0., 10.])
        self.task = Task(target_pos=self.target_pos)
        self.agent = DDPGAgent(self.task)

    def test_train(self):
        for i_episode in range(1, self.num_episodes+1):
            state = self.agent.reset_episode() # start a new episode
            self.agent.score=0.0
            while True:
                action = self.agent.act(state)
                next_state, reward, done = self.task.step(action)
                self.agent.step(action,reward, next_state,done)
                state = next_state
                self.agent.score += reward
                self.agent.best_score = max(self.agent.best_score, self.agent.score)
                self.agent.worst_score = min(self.agent.worst_score, self.agent.score)
                if done:
                    print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                        i_episode, self.agent.score, self.agent.best_score), end="")
                    break
            sys.stdout.flush()