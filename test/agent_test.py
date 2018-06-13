import sys
import unittest
from task2 import Task
import numpy as np
from agents.agent import DDPGAgent

class DDPGTestCase(unittest.TestCase):

    def setUp(self):
        self.num_episodes = 500

        target_pos = np.array([0., 0., 10.])
        init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
        init_velocities = np.array([0., 0., 0.])  # initial velocities
        init_angle_velocities = np.array([0., 0., 0.])  # initial angle velocities

        self.task = Task(init_pose=init_pose,init_velocities=init_velocities,
                         init_angle_velocities=init_angle_velocities,
                         target_pos=target_pos)
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
                angular_v = self.task.sim.angular_v.mean()
                pos = self.task.sim.pose[:3]
                velocity = self.task.sim.v.mean()
                if done:
                    print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}) (worst = {:7.3f}) (reward = {:7.3f})".format(
                        i_episode, self.agent.score, self.agent.best_score,self.agent.worst_score,reward), end="")
                    break
            sys.stdout.flush()