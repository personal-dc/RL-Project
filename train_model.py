import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# class DiscretizedCarRacing(gym.Env):
#     def __init__(self):
#         # Create the original CarRacing environment
#         self.env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
#         self.observation_space = self.env.observation_space
#         self.action_space = spaces.Discrete(5)

#     def step(self, action):
#         action = int(action)
#         return self.env.step(action)

#     def render(self):
#         return self.env.render()

#     def close(self):
#         self.env.close()
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

# env = DiscretizedCarRacing()
env = gym.make('CarRacing-v3', render_mode='rgb_array')
# Wrap the environment with DummyVecEnv (Stable-Baselines3 requires vectorized environments)
env = DummyVecEnv([lambda: env])

# Define PPO model
model = PPO('CnnPolicy', env, verbose=1)
# model = DQN('CnnPolicy', env, verbose=1)


# Train the model
model.learn(total_timesteps=1000000)  # You can increase the timesteps for better training

# Save the model
model.save("ppo_carracing_more")

# Evaluate the model
obs = env.reset()
for _ in range(1000):  # Test for 1000 steps (you can adjust based on your needs)
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()

# Close the environment after testing
env.close()
