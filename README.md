# Nurse Scheduling Optimization with Reinforcement Learning

## Project Overview
This project applies Reinforcement Learning (RL) to solve a nurse scheduling optimization problem. The goal is to handle nurse absences effectively while minimizing disruptions and maximizing scheduling efficiency.

The scheduling environment is modeled as a Markov Decision Process (MDP), where:
- **States** represent the current scheduling status of nurses.
- **Actions** include options such as swapping shifts, assigning overtime, or making no changes.
- **Rewards** incentivize efficient scheduling by assigning positive rewards for successful actions and penalties for disruptions.

## Features
- **Custom RL Environment**: Implements an OpenAI Gym-style environment to simulate nurse scheduling dynamics.
- **Probabilistic Absence Simulation**: Randomly generates nurse absences based on a specified probability.
- **Multiple Actions**: Includes actions for swapping shifts, assigning overtime, and making no changes.
- **Reward Tracking**: Tracks and visualizes rewards across episodes to evaluate training performance.
- **Customizability**: Allows fine-tuning of absence probabilities, rewards, and scheduling constraints.

## File Structure
- `main.ipynb`: Jupyter Notebook containing the implementation of the RL model, environment, and analysis.
- `env.py`: Defines the custom nurse scheduling environment.
- `callbacks.py`: Contains the RewardTrackingCallback for monitoring training rewards.
- `README.md`: Documentation for the project (this file).

## Requirements
- Python 3.8+
- Required libraries:
  - numpy
  - matplotlib
  - gymnasium
  - stable-baselines3

## How It Works
1. **Environment Setup**:
   The `RlSchedEnv` environment models nurse assignments over multiple days. It supports actions to:
   - Swap shifts with another nurse.
   - Assign overtime to a resting nurse.
   - Leave the schedule unchanged.

2. **Reinforcement Learning Model**:
   A Proximal Policy Optimization (PPO) model is trained on the environment to learn optimal scheduling policies. The model is configured to handle:
   - Dynamic nurse absences.
   - Reward feedback based on scheduling efficiency.

3. **Training**:
   Training is conducted over multiple episodes, with each episode simulating several scheduling periods.

4. **Evaluation**:
   The rewards are visualized to assess the model’s performance as it converges.

## Example Usage
To train the model and track rewards:
1. Instantiate the environment:
   ```python
   env = RlSchedEnv(data_path, solution_path)
   ```
2. Train the model:
   ```python
   from stable_baselines3 import PPO
   from callbacks import RewardTrackingCallback

   reward_callback = RewardTrackingCallback()
   model = PPO("MlpPolicy", env)
   model.learn(total_timesteps=15000, callback=reward_callback)
   ```
3. Visualize training rewards:
   ```python
   import matplotlib.pyplot as plt
   import numpy as np

   rewards = np.load("training_rewards.npy")
   plt.plot(rewards)
   plt.title("Rewards Over Episodes")
   plt.xlabel("Episode")
   plt.ylabel("Reward")
   plt.show()
   ```

## Customization
- **Rewards**: Adjust the `self.rewards` dictionary in `RlSchedEnv` to customize incentives.
- **Absence Probability**: Modify the `ABSENCE_P` parameter to simulate varying levels of nurse absences.
- **Actions**: Extend or modify the action space in `RlSchedEnv` to add new scheduling options.

## Future Enhancements
- Add support for more complex scheduling constraints (e.g., minimum rest periods, skill requirements).
- Integrate real-world data for validation and benchmarking.
- Implement multi-agent RL to handle multi-hospital scheduling scenarios.

## References
- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
