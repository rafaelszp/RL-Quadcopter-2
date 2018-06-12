import unittest
from task import Task
import numpy as np
from agents.agent import QuadcopterAgent

class TestSuite(unittest.TestCase):


    def setUp(self):
        runtime = 5.  # time limit of the episode
        init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
        init_velocities = np.array([0., 0., 0.])  # initial velocities
        init_angle_velocities = np.array([0., 0., 0.])  # initial angle velocities
        self.task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
        self.agent = QuadcopterAgent(self.task)


    def test_stateShouldHave12Elements(self):
        action = self.agent.act()
        expected_size = 12*self.task.action_repeat
        next_state,reward,done = self.task.step(action)
        self.assertEqual(next_state.shape[0],expected_size,'Expected size is correct')