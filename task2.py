import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        #sigmoid = lambda x: 1./(np.exp(-x))
        delta_x = np.tanh(abs(self.sim.pose[0] - self.target_pos[0]))
        delta_y = np.tanh(abs(self.sim.pose[1] - self.target_pos[1]))
        delta_z = np.tanh(abs(self.sim.pose[2] - self.target_pos[2]))

        target_z = self.target_pos[2]
        pos_z = self.sim.pose[2]

        distance = delta_y+delta_x+delta_z

        reward = 1-distance*0.04
        
        #reward = - min(delta_z,20)
        #reward = 1- 0.003*distance
        #reward = np.tanh(reward)
        #if pos_z>=target_z:

        # #print(delta_z,reward,self.sim.pose[2] )
        # if(reward>0):
        #     reward = max(reward,-10)
        # if reward < 0:
        #     reward = min(reward,10)

        #penalty_multiplier = .001
        #reward = reward - (1-penalty_multiplier * delta_y) - (1-penalty_multiplier * delta_x)
        #reward = np.tanh(reward) - 0.03*delta_z
        #reward = np.tanh(1 - 0.003 * (abs(self.sim.pose[2] - self.target_pos[2]))).sum()

        return reward



    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(np.concatenate((self.sim.pose, self.sim.v,self.sim.angular_v)))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose,self.sim.v,self.sim.angular_v] * self.action_repeat)
        return state