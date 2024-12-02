import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class SchedulingEnv(gym.Env):
    def __init__(self, data_path, solution_path, overtime_p=1.0):
        super(SchedulingEnv, self).__init__()

        # Load the data and solution
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        with open(solution_path, 'r') as f:
            self.solution = json.load(f)
        
        # Hyperparameter
        self.overtime_p = overtime_p  # Probability of a nurse accepting overtime
        
        # Initialize states
        self.current_schedule = self.solution['schedule']
        self.nurses = self.data['nurses']
        self.shifts = self.data['shifts']
        self.time_steps = len(self.shifts)  # Assuming each shift corresponds to a time step
        self.current_step = 0

        # Action space: 0 = no action (unchanged), 1 = swap, 2 = overtime
        self.action_space = spaces.Discrete(3)

        # Observation space: Here we keep the number of nurses, shifts, etc.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.nurses), len(self.shifts)), dtype=np.int32
        )

    def reset(self):
        self.current_schedule = self.solution['schedule']  # reset to the initial schedule
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        # Create the current state representation
        state = np.zeros((len(self.nurses), len(self.shifts)), dtype=np.int32)

        # Fill the state matrix with current assignments (1 = assigned, 0 = not assigned)
        for nurse_idx, assignments in enumerate(self.current_schedule):
            for shift in assignments:
                state[nurse_idx, shift] = 1
        return state.flatten()

    def step(self, action):
        reward = 0
        done = False

        # Action Handling
        if action == 0:
            reward = -1  # No change (penalty for not addressing absence)
        elif action == 1:
            reward = self._perform_swap()
        elif action == 2:
            reward = self._offer_overtime()

        # Increment time step
        self.current_step += 1
        if self.current_step >= self.time_steps:
            done = True  # end of schedule window

        return self._get_obs(), reward, done, {}

    def _perform_swap(self):
        # Randomly select a nurse to swap assignments (simplified)
        nurse_1, nurse_2 = random.sample(range(len(self.nurses)), 2)
        shift_1 = random.choice(self.current_schedule[nurse_1])
        shift_2 = random.choice(self.current_schedule[nurse_2])

        # Swap the shifts
        self.current_schedule[nurse_1].remove(shift_1)
        self.current_schedule[nurse_1].append(shift_2)
        self.current_schedule[nurse_2].remove(shift_2)
        self.current_schedule[nurse_2].append(shift_1)

        return -1  # No reward for the swap itself

    def _offer_overtime(self):
        # Randomly offer overtime to a nurse
        overtime_acceptance = np.random.rand() < self.overtime_p

        if overtime_acceptance:
            return -2  # Overtime penalty (cost)
        else:
            return -3  # Understaffing penalty (cost)


if __name__ == '__main__':
    # Create environment and wrap it in DummyVecEnv for training
    env = SchedulingEnv(data_path='data/W1-01.json', solution_path='solutions/sol-W1-01.json', overtime_p=1.0)
    env = DummyVecEnv([lambda: env])

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Save the model after training
    model.save("scheduling_ppo_model")

    # Load the trained model
    model = PPO.load("scheduling_ppo_model")

    # Test the model
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(f"Action: {action}, Reward: {rewards}")
        if done:
            break